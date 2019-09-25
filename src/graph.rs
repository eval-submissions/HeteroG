use protobuf::Message;
use crate::proto::{graph::GraphDef, node_def::NodeDef, attr_value};
use std::collections::BTreeMap;
use std::fmt::Write;
use std::convert::TryInto;
use crate::strategy::Strategy;

pub struct Graph<NEX: Default, TEX: Default> {
    pub nodes: Vec<Node<NEX, TEX>>, // This vector is partial ordered: inputs are guarenteed to appear ealier than descendents
    pub name_dict: std::collections::BTreeMap<String, usize>
}

impl<NEX: Default, TEX: Default> Graph<NEX, TEX> {
    pub fn new<'a, T: IntoIterator<Item=&'a NodeDef>>(iter: T) -> Box<Self> {
        let mut g = Box::new(Graph { nodes: Vec::new(), name_dict: BTreeMap::new() });

        // no always optimal, but good enough since the input is actually mostly ordered
        let mut queue: std::collections::VecDeque<_> = iter.into_iter().collect();
        'outer: while let Some(node_def) = queue.pop_front() {
            for input in node_def.input.iter() {
                let input = if input.starts_with('^') {
                    &input[1..]
                } else {
                    parse_input(input).0
                };
                if !g.name_dict.contains_key(input) {
                    queue.push_back(node_def);
                    continue 'outer;
                }
            }

            let node = Node::new(&g, node_def.clone(), "".to_string());
            g.name_dict.insert(node.raw_node.name.clone(), g.nodes.len());
            g.nodes.push(node);
        }

        g
    }

    /// setup the replicas and links. Note that auxiliary nodes are already there by strategies.
    pub fn compile(&mut self, target: &mut Target) {
        for node in self.nodes.iter_mut() {
            node.compile(target)
        }
    }
}

pub struct Node<NEX: Default, TEX: Default> {
    pub graph: *const Graph<NEX, TEX>,
    pub raw_node: NodeDef,
    pub device: String,
    pub controls: Vec<usize>, // TODO: more consideration for control dependencies that added aux nodes
    pub inputs: Vec<(usize, usize)>, // nodeid, index
    pub outputs: Vec<Tensor<NEX, TEX>>,
    pub replicas: Vec<(usize, String)>, // deviceid, name. no element: not determined; single element: just place; multiple elements: currently there must be exactly one replica each device

    pub extra: NEX
}

impl<NEX: Default, TEX: Default> Node<NEX, TEX> {
    pub fn new(graph: &Graph<NEX, TEX>, raw_node: NodeDef, device: String) -> Self {
        let mut inputs = vec![];
        let mut controls = vec![];

        for input in raw_node.input.iter() {
            if input.starts_with('^') {
                controls.push(graph.name_dict[&input[1..]])
            } else {
                let (name, index) = parse_input(input);
                let id = graph.name_dict[name];
                inputs.push((id, index))
            }
        }

        Self {
            graph, raw_node, device,
            controls, inputs,
            outputs: vec![],
            replicas: vec![],
            extra: Default::default()
        }
    }

    // TODO: use RC+Cell to get rid of the unsafe? It's an recursive reference so we need a mannual destroy function
    #[allow(clippy::mut_from_ref)]
    pub fn graph(&self) -> &mut Graph<NEX, TEX> {
        unsafe { &mut *(self.graph as *mut Graph<NEX, TEX>) }
    }

    #[allow(clippy::mut_from_ref, clippy::cast_ref_to_mut)]
    pub fn get_output(&self, index: usize) -> &mut Tensor<NEX, TEX> {
        // TODO: use Interior mutable for outputs?
        let mutable = unsafe { &mut *(self as *const Node<NEX, TEX> as *mut Node<NEX, TEX>) };

        while mutable.outputs.len() <= index {
            mutable.outputs.push(Tensor::new(mutable, mutable.outputs.len()))
        }

        &mut mutable.outputs[index]
    }

    pub fn replicated(&self) -> Option<bool> {
        match self.replicas.len() {
            0 => None,
            1 => Some(false),
            _ => Some(true)
        }
    }

    /// add an edited node into the target. Requires all inputs to be compiled first
    fn compile(&mut self, target: &mut Target) {
        if self.replicas.len() > 1 {
            for (device_id, name) in self.replicas.iter() {
                let mut node = self.raw_node.clone();
                node.name = name.clone();
                node.device = target.devices[*device_id].clone();
                node.input = self.inputs.iter().map(|(node_id, index)| {
                    let input = &self.graph().nodes[*node_id];
                    input.get_output(*index).get_replicated(*device_id)
                }).collect();
                for node_id in self.controls.iter() {
                    let dep_node = &self.graph().nodes[*node_id];
                    let dependency = if dep_node.replicated().unwrap() {
                        &dep_node.replicas[*device_id].1
                    } else {
                        &dep_node.replicas[0].1
                    };
                    node.input.push(format!("^{}", dependency))
                }
                target.pb.node.push(node)
            }
        } else {
            let mut node = self.raw_node.clone();
            node.device = target.devices[self.replicas[0].0].clone();
            node.input = self.inputs.iter().map(|(node_id, index)| {
                let input = &self.graph().nodes[*node_id];
                input.get_output(*index).get_aggregated()
            }).collect();
            for node_id in self.controls.iter() {
                for (_, replica) in &self.graph().nodes[*node_id].replicas {
                    node.input.push(format!("^{}", replica))
                }
            }
            target.pb.node.push(node)
        }
    }
}

pub struct Tensor<NEX: Default, TEX: Default> {
    pub node: *const Node<NEX, TEX>,
    pub index: usize,

    pub replicated: Option<Box<dyn Fn(usize) -> String>>, // should be provided by strategy
    pub aggregated: Option<String>, // should be provided by strategy

    pub extra: TEX,
}

impl<NEX: Default, TEX: Default> Tensor<NEX, TEX> {
    pub fn new(node: &Node<NEX, TEX>, index: usize) -> Self {
        Tensor { node, index,
            replicated: None,
            aggregated: None,
            extra: Default::default()
        }
    }

    pub fn original_name(&self) -> String {
        if self.index == 0 {
            self.node().raw_node.name.clone()
        } else {
            format!("{}:{}", self.node().raw_node.name, self.index)
        }
    }

    pub fn node(&self) -> &Node<NEX, TEX> {
        unsafe { &*self.node }
    }

    pub fn get_replicated(&self, device_id: usize) -> String {
        self.replicated.as_ref().unwrap()(device_id)
    }

    pub fn get_aggregated(&self) -> String {
        self.aggregated.clone().unwrap()
    }

    pub fn get_shape(&self) -> Vec<usize> {
        if let Some(attr_value::AttrValue_oneof_value::list(list)) = &self.node().raw_node.attr["_output_shapes"].value {
            let dims = &list.shape[self.index].dim;
            dims.iter().map(|x| x.size.try_into().unwrap()).collect()
        } else {
            panic!("no shape information")
        }
    }
}

pub struct Target {
    pub pb: GraphDef,
    pub devices: Box<[String]>,
    pub bandwidth_matrix: Box<[f32]> // the i*n+j element is the bandwidth (in 1024 bytes per second) from i to j
}

impl Target {
    pub fn new(pb: GraphDef, devices: Box<[String]>) -> Self {
        let n = devices.len();
        Target { pb, devices, bandwidth_matrix: vec![1.0; n * n].into_boxed_slice() }
    }
}

fn parse_input(x: &str) -> (&str, usize) {
    match x.find(':') {
        Some(i) => (&x[..i], x[i+1..].parse().unwrap()),
        None => (x, 0)
    }
}
