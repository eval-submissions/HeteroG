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
    pub fn new<'a, T: IntoIterator<Item=&'a NodeDef>>(iter: T) -> Box<Self> {
        let mut g = Box::new(Graph { nodes: Vec::new(), name_dict: BTreeMap::new() });

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

    /// setup the replicas and links. Note that auxiliary nodes are already there by strategies.
    pub fn compile(&mut self, target: &mut Target) {
        for node in self.nodes.iter_mut() {
            node.compile(target)
        }
    }
}

pub struct Node {
    pub graph: *const Graph,
    pub raw_node: NodeDef,
    pub device: String,
    pub controls: Vec<usize>, // TODO: more consideration for control dependencies that added aux nodes
    pub inputs: Vec<(usize, usize)>, // nodeid, index
    pub outputs: Vec<Tensor>,

    pub replicas: Vec<(usize, String)>, // deviceid, name. no element: not determined; single element: just place; multiple elements: currently there must be exactly one replica each device
    pub compiled: bool
}

impl Node {
    pub fn new(graph: &Graph, raw_node: NodeDef, device: String) -> Self {
        Node {
            graph, raw_node, device,
            controls: vec![],
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

    pub fn get_output(&self, index: usize) -> &mut Tensor {
        // TODO: use Interior mutable for outputs?
        let mutable = unsafe { &mut *(self as *const Node as *mut Node) };

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

pub struct Tensor {
    pub node: *const Node,
    pub index: usize,

    pub replicated: Option<Box<dyn Fn(usize) -> String>>, // should be provided by strategy
    pub aggregated: Option<String> // should be provided by strategy
}

impl Tensor {
    pub fn new(node: &Node, index: usize) -> Self {
        Tensor { node, index,
            replicated: None,
            aggregated: None
        }
    }

    pub fn original_name(&self) -> String {
        if self.index == 0 {
            self.node().raw_node.name.clone()
        } else {
            format!("{}:{}", self.node().raw_node.name, self.index)
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
