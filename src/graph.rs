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
    pub fn new(nodes: &[NodeDef]) -> Box<Self> {
        let mut g = Box::new(Graph { nodes: Vec::with_capacity(nodes.len()), name_dict: BTreeMap::new() });

        // no always optimal, but good enough since the input is actually mostly ordered
        let mut queue: std::collections::VecDeque::<_> = nodes.iter().collect();
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

            let node = Node::new(&g, node_def.clone());
            g.name_dict.insert(node.raw_node.name.clone(), g.nodes.len());
            g.nodes.push(node);
        }

        g
    }

    /// setup the replicas and links. Note that auxiliary nodes are already there by strategies.
    /// The only required fields is `replicas`. All other decisions are optional.
    pub fn compile(&mut self, target: &mut Target) {
        for node in self.nodes.iter_mut() {
            node.compile(target)
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
pub enum ReplicationType { Cache, Split }

pub struct Node<NEX: Default, TEX: Default> {
    pub graph: *const Graph<NEX, TEX>,
    pub raw_node: NodeDef,
    pub controls: Vec<usize>, // TODO: more consideration for control dependencies that added aux nodes
    pub inputs: Vec<(usize, usize)>, // nodeid, index
    pub outputs: Vec<Tensor<NEX, TEX>>,

    pub replicas: Vec<(usize, String)>, // deviceid, name. This should be filled by strategies.
    pub input_replication_types: Vec<ReplicationType>, // the replication type of each input. This should be filled by strategies.

    pub extra: NEX
}

impl<NEX: Default, TEX: Default> Node<NEX, TEX> {
    pub fn new(graph: &Graph<NEX, TEX>, raw_node: NodeDef) -> Self {
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
            graph, raw_node, controls, inputs,
            outputs: vec![], replicas: vec![],
            input_replication_types: vec![],
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

    // if any of the inputs are splitted.
    pub fn is_splitted(&self) -> bool {
        self.input_replication_types.iter().any(|x| x == &ReplicationType::Split)
    }

    /// add an edited node into the target. Requires all inputs to be compiled first
    fn compile(&mut self, target: &mut Target) {
        for (device_id, name) in self.replicas.iter() {
            // 1. setup basic node info
            let mut node = self.raw_node.clone();
            node.name = name.clone();
            node.device = target.devices[*device_id].clone();
            let mut attr = attr_value::AttrValue::new();
            attr.value = Some(attr_value::AttrValue_oneof_value::s(self.raw_node.name.as_bytes().to_vec()));
            node.attr.insert("_tge_origin".into(), attr);

            // 2. link inputs
            node.input = self.inputs.iter().zip(&self.input_replication_types).map(|((node_id, index), reptype)| {
                let input_tensor = &mut self.graph().nodes[*node_id].get_output(*index);
                match reptype {
                    ReplicationType::Cache => input_tensor.get_cache(*device_id, target),
                    ReplicationType::Split => input_tensor.get_split(*device_id, target)
                }
            }).collect();
            
            // 3. add control dependencies
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
        // } else {
        //     let mut node = self.raw_node.clone();
        //     node.device = target.devices[self.replicas[0].0].clone();
        //     node.input = self.inputs.iter().map(|(node_id, index)| {
        //         let input = &self.graph().nodes[*node_id];
        //         input.get_output(*index).get_aggregated()
        //     }).collect();
        //     let mut attr = attr_value::AttrValue::new();
        //     attr.value = Some(attr_value::AttrValue_oneof_value::s(self.raw_node.name.as_bytes().to_vec()));
        //     node.attr.insert("_tge_origin".into(), attr);
        //     for node_id in self.controls.iter() {
        //         for (_, replica) in &self.graph().nodes[*node_id].replicas {
        //             node.input.push(format!("^{}", replica))
        //         }
        //     }
        //     target.pb.node.push(node)
        // }
    }
}

pub struct Tensor<NEX: Default, TEX: Default> {
    pub node: *const Node<NEX, TEX>,
    pub index: usize,

    pub cache: Vec<String>, // should be provided by strategy
    pub split: Vec<String>, // should be provided by strategy

    pub extra: TEX,
}

impl<NEX: Default, TEX: Default> Tensor<NEX, TEX> {
    pub fn new(node: &Node<NEX, TEX>, index: usize) -> Self {
        Tensor { node, index, cache: vec![], split: vec![], extra: TEX::default() }
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

    pub fn get_shape(&self) -> Vec<usize> {
        if let Some(attr_value::AttrValue_oneof_value::list(list)) = &self.node().raw_node.attr["_output_shapes"].value {
            // sucks: the output shape of BroadcastGradientArgs is always unknown even if inputs are fixed
            // and ops like `Sum` (requires the dimension to sum along with) and `Fill` operates differenct with different input
            list.shape[self.index].dim.iter().map(|x| x.size.try_into().ok()).collect::<Option<_>>().unwrap_or_else(Vec::new)
        } else {
            panic!("no shape information")
        }
    }

    pub fn get_cache(&mut self, device_id: usize, target: &mut Target) -> String {
        if self.cache.is_empty() {
            if self.node().replicated().unwrap() {
                // using PS
            } else {
                let name = format!("{}:{}", self.node().replicas[0].1, self.index);
                self.cache.extend((0..target.devices.len()).map(|_| name.clone()));
            }
        }

        self.cache[device_id].clone()
    }

    pub fn get_split(&mut self, device_id: usize, target: &mut Target) -> String {

        self.split[device_id].clone()
    }
}

pub struct Target {
    pub pb: GraphDef,
    pub devices: Box<[String]>
    // TODO: add network topology information
}

impl Target {
    pub fn new(pb: GraphDef, devices: Box<[String]>) -> Self {
        let n = devices.len();
        Target { pb, devices }
    }
}

fn parse_input(x: &str) -> (&str, usize) {
    match x.find(':') {
        Some(i) => (&x[..i], x[i+1..].parse().unwrap()),
        None => (x, 0)
    }
}
