use protobuf::Message;
use crate::proto::{graph::GraphDef, node_def::NodeDef};
use std::collections::BTreeMap;
use std::fmt::Write;
use crate::strategy::Strategy;

pub struct Graph {
    pub nodes: Vec<Node>,
    pub name_dict: std::collections::BTreeMap<String, usize>
}

impl Graph {
    /// setup the replicas and links. Note that auxiliary nodes are already there by strategies.
    pub fn compile(&mut self, target: &mut Target) {
        for node in self.nodes.iter_mut() {
            node.compile(target)
        }
    }
}

impl<'a> std::iter::FromIterator<&'a NodeDef> for Graph {
    fn from_iter<T: IntoIterator<Item=&'a NodeDef>>(iter: T) -> Self {
        let mut g = Graph { nodes: Vec::new(), name_dict: BTreeMap::new() };

        for node_def in iter {
            let mut node = Node::new(&g, node_def.clone(), "".into());
            g.name_dict.insert(node.raw_node.name.clone(), g.nodes.len());
            g.nodes.push(node);
        }

        for node in g.nodes.iter_mut() {
            node.link_inputs()
        }

        g
    }
}

pub enum ReplicationMethod { undefined, copy, cache, split, sum }

pub struct Node {
    pub graph: *const Graph,
    pub raw_node: NodeDef,
    pub device: String,
    pub inputs: Vec<(usize, usize)>, // nodeid, index
    pub outputs: Vec<Tensor>,

    pub replicas: Vec<(usize, String)>, // deviceid, name. no element: not determined; single element: just place; multiple elements: currently there must be exactly one replica each device
    pub compiled: bool
}

impl Node {
    pub fn new(graph: &Graph, raw_node: NodeDef, device: String) -> Self {
        Node {
            graph, raw_node, device,
            inputs: vec![],
            outputs: vec![],
            replicas: vec![],
            compiled: false
        }
    }

    // TODO: use RC to get this right
    pub fn graph(&self) -> &mut Graph {
        unsafe { &mut *(self.graph as *mut Graph) }
    }

    pub fn link_inputs(&mut self) {
        self.inputs = self.raw_node.input.iter().map(|input| {
            let (name, index) = parse_input(input);
            let id = self.graph().name_dict[name];
            (id, index)
        }).collect();
    }

    pub fn get_output(&self, index: usize) -> &mut Tensor {
        // TODO: use Interior mutable for outputs?
        let mutable = unsafe { &mut *(self as *const Node as *mut Node) };

        while mutable.outputs.len() <= index {
            mutable.outputs.push(Tensor::new(mutable, mutable.outputs.len()))
        }

        &mut mutable.outputs[index]
    }

    /// recursively compile ancestor tree
    fn compile(&mut self, target: &mut Target) {
        if self.compiled {
            return
        }

        for (node_id, index) in self.inputs.iter() {
            self.graph().nodes[*node_id].compile(target)
        }

        if self.replicas.len() > 1 {
            for (device_id, name) in self.replicas.iter() {
                let mut node = self.raw_node.clone();
                node.name = name.clone();
                node.input = self.inputs.iter().map(|(node_id, index)| {
                    let input = &self.graph().nodes[*node_id];
                    input.get_output(*index).get_replicated(*device_id)
                }).collect();
            }
        } else {
            let mut node = self.raw_node.clone();
            node.input = self.inputs.iter().map(|(node_id, index)| {
                let input = &self.graph().nodes[*node_id];
                input.get_output(*index).get_aggregated()
            }).collect();
        }

        self.compiled = true
    }
}

// replicability: deducted from inputs ability
// requirement: requested by the output
// state: by the statue of the node
// resolve: choose the merge/split/all-reduce implementation using the above condition and strategy
pub struct Tensor {
    pub node: *const Node,
    pub index: usize,

    pub replicability: Vec<ReplicationMethod>, // how it can be replicated when needed
    pub required: ReplicationMethod, // how the child want it to be
    pub provided: ReplicationMethod, // how it currently is

    pub replicated: Option<Box<dyn Fn(usize) -> String>>, // should be provided by strategy
    pub aggregated: Option<String> // should be provided by strategy
}

impl Tensor {
    pub fn new(node: &Node, index: usize) -> Self {
        Tensor { node, index,
            replicability: vec![],
            required: ReplicationMethod::undefined,
            provided: ReplicationMethod::undefined,
            replicated: None,
            aggregated: None
        }
    }

    pub fn node(&self) -> &Node {
        unsafe { &*self.node }
    }

    pub fn get_replicated(&self, device_id: usize) -> String {
        self.replicated.as_ref().unwrap()(device_id)
    }

    pub fn get_aggregated(&self) -> String {
        self.aggregated.clone().unwrap()
    }
}

pub struct Target {
    pub pb: GraphDef,
    pub devices: Box<[String]>
}

impl Target {
    pub fn new(pb: GraphDef, devices: &[&str]) -> Self {
        Target { pb, devices: devices.iter().map(|x| (*x).to_owned()).collect::<Vec<_>>().into_boxed_slice() }
    }
}

fn parse_input(x: &str) -> (&str, usize) {
    match x.find(':') {
        Some(i) => (&x[..i], x[i+1..].parse().unwrap()),
        None => (x, 0)
    }
}
