use protobuf::Message;
use crate::proto::{graph::GraphDef, node_def::NodeDef};
use std::collections::BTreeMap;
use std::fmt::Write;
use crate::strategy::Strategy;

pub struct Graph<NEX: Default, TEX: Default> {
    pub nodes: Vec<Node<NEX, TEX>>,
    pub name_dict: std::collections::BTreeMap<String, usize>
}

impl<NEX: Default, TEX: Default> Graph<NEX, TEX> {
    pub fn new<'a, T: IntoIterator<Item=&'a NodeDef>>(iter: T) -> Box<Self> {
        let mut g = Box::new(Graph { nodes: Vec::new(), name_dict: BTreeMap::new() });

        for node_def in iter {
            let node = Node::new(&g, node_def.clone(), "".into());
            g.name_dict.insert(node.raw_node.name.clone(), g.nodes.len());
            g.nodes.push(node);
        }

        for node in g.nodes.iter_mut() {
            node.link_inputs()
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
    pub compiled: bool,

    pub extra: NEX
}

impl<NEX: Default, TEX: Default> Node<NEX, TEX> {
    pub fn new(graph: &Graph<NEX, TEX>, raw_node: NodeDef, device: String) -> Self {
        Self {
            graph, raw_node, device,
            controls: vec![],
            inputs: vec![],
            outputs: vec![],
            replicas: vec![],
            compiled: false,
            extra: Default::default()
        }
    }

    // TODO: use RC to get this right
    pub fn graph(&self) -> &mut Graph<NEX, TEX> {
        unsafe { &mut *(self.graph as *mut Graph<NEX, TEX>) }
    }

    pub fn link_inputs(&mut self) {
        for input in self.raw_node.input.iter() {
            if input.starts_with('^') {
                self.controls.push(self.graph().name_dict[&input[1..]])
            } else {
                let (name, index) = parse_input(input);
                let id = self.graph().name_dict[name];
                self.inputs.push((id, index))
            }
        }
    }

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

    /// recursively compile ancestor tree
    fn compile(&mut self, target: &mut Target) {
        if self.compiled {
            return
        }

        for (node_id, _) in self.inputs.iter() {
            self.graph().nodes[*node_id].compile(target)
        }

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

        self.compiled = true
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
        unimplemented!()
    }
}

pub struct Target {
    pub pb: GraphDef,
    pub devices: Box<[String]>
}

impl Target {
    pub fn new(pb: GraphDef, devices: Box<[String]>) -> Self {
        Target { pb, devices }
    }
}

fn parse_input(x: &str) -> (&str, usize) {
    match x.find(':') {
        Some(i) => (&x[..i], x[i+1..].parse().unwrap()),
        None => (x, 0)
    }
}
