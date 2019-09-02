use protobuf::Message;
use crate::proto::{graph::GraphDef, node_def::NodeDef};
use std::collections::BTreeMap;
use std::fmt::Write;

pub struct Graph {
    nodes: Vec<Node>,
    target: Option<Target>,
    name_dict: std::collections::BTreeMap<String, usize>
}

impl Graph {
    pub fn replicate_all(&mut self, target: Target) {
        self.target = Some(target);

        for node in self.nodes.iter_mut() {
            node.replicate_recursive()
        }
    }
}

impl<'a> std::iter::FromIterator<&'a NodeDef> for Graph {
    fn from_iter<T: IntoIterator<Item=&'a NodeDef>>(iter: T) -> Self {
        let mut g = Graph { nodes: Vec::new(), target: None, name_dict: BTreeMap::new() };

        for node_def in iter {
            let mut node = Node::new(&g, node_def.clone(), "".into());
            g.name_dict.insert(node.raw_node.name.clone(), g.nodes.len() - 1);
            g.nodes.push(node);
        }

        for node in g.nodes.iter_mut() {
            node.link_inputs()
        }

        g
    }
}

pub enum Replication {
    Undefined,
    Singleton(String),
    Replicas(Vec<String>)
}

pub enum ReplicationMethod { copy, cache, split, sum, temp } // temp means not decided yet since we do not add the inference rules

pub struct Node {
    graph: *const Graph,
    raw_node: NodeDef,
    device: String,
    inputs: Vec<(usize, usize)>, // nodeid, index
    outputs: Vec<Tensor>,

    replication: Replication
}

impl Node {
    fn new(graph: &Graph, raw_node: NodeDef, device: String) -> Self {
        Node {
            graph, raw_node, device,
            inputs: vec![],
            outputs: vec![],
            replication: Replication::Undefined
        }
    }

    // TODO: use RC to get this right
    fn graph(&self) -> &mut Graph {
        unsafe { &mut *(self.graph as *mut Graph) }
    }

    fn link_inputs(&mut self) {
        self.inputs = self.raw_node.input.iter().map(|input| {
            let (name, index) = parse_input(input);
            let id = self.graph().name_dict[name];
            (id, index)
        }).collect();
    }

    fn replicate_recursive(&mut self) {
        if let Replication::Undefined = self.replication {} else {
            return
        }

        let new_replication;
        let target = self.graph().target.as_mut().unwrap();

        for (id, _) in &self.inputs {
            let mut input = &mut self.graph().nodes[*id];
            input.replicate_recursive();
        }

        // temporary logic
        if self.raw_node.op == "VariableV2" { // variables cannot be replicated but its output can be cached
            target.pb.node.push(self.raw_node.clone());
            new_replication = Replication::Singleton(self.raw_node.name.clone());
            let x = &mut self.get_output(0);
            x.method = ReplicationMethod::cache;
        } else if self.raw_node.op == "Placeholder" { // Placeholder cannot be replicated but its outputs can be splited
            target.pb.node.push(self.raw_node.clone());
            new_replication = Replication::Singleton(self.raw_node.name.clone());
            let x = &mut self.get_output(0);
            x.method = ReplicationMethod::split;
        } else { // general case, assume pure function, simply duplicate the operator using inputs on the same device
            let replicas = (0..target.devices.len()).map(|id| {
                let mut x = self.raw_node.clone();

                // setup name
                write!(&mut x.name, "/replica_{}", id).unwrap();
                let name = x.name.clone();

                // setup device
                x.device = target.devices[id].clone();

                // setup inputs
                x.input.clear();
                for (node_id, index) in &self.inputs {
                    let node = &self.graph().nodes[*node_id];
                    let tensor_name = node.get_output(*index).get_replicated(id);
                    x.input.push(tensor_name);
                }

                target.pb.node.push(x);
                name
            }).collect();
            new_replication = Replication::Replicas(replicas);
        }

        self.replication = new_replication;
    }

    fn get_output(&self, index: usize) -> &mut Tensor {
        // TODO: use Interior mutable for outputs?
        let mutable = unsafe { &mut *((self as *const Node) as *mut Node) };

        while mutable.outputs.len() <= index {
            mutable.outputs.push(Tensor::new(mutable, mutable.outputs.len()))
        }

        &mut mutable.outputs[index]
    }
}

// replicability: deducted from inputs ability
// requirement: requested by the output
// state: by the statue of the node
// resolve: choose the merge/split/all-reduce implementation using the above condition and strategy
pub struct Tensor {
    node: *const Node,
    index: usize,
    method: ReplicationMethod,
    replicated: Option<Box<dyn Fn(usize) -> String>>
}

impl Tensor {
    fn new(node: &Node, index: usize) -> Self {
        Tensor { node, index, method: ReplicationMethod::cache, replicated: None }
    }

    fn node(&self) -> &Node {
        unsafe { &*self.node }
    }

    fn get_replicated(&mut self, device_id: usize) -> String {
        if let Replication::Replicas(replicas) = &self.node().replication {
            let replicas = replicas.clone();
            let index = self.index;
            self.replicated = Some(Box::new(move |id| format!("{}:{}", replicas[id], index)))
        }

        match self.method {
            ReplicationMethod::cache => self.replicate_cache(),
            ReplicationMethod::split => unimplemented!(),
            _ => unreachable!()
        }

        self.replicated.as_ref().unwrap()(device_id)
    }

    fn replicate_cache(&mut self) {
        let target = self.node().graph().target.as_mut().unwrap();
        for (id, device) in target.devices.iter().enumerate() {
            let mut identity = NodeDef::new();
            identity.name = format!("{}/aux_identity_{}", self.node().raw_node.name, id);
            identity.op = "Identity".into();
            identity.device = device.into();
            if let Replication::Singleton(x) = &self.node().replication {
                identity.input.push(x.clone())
            } else {
                panic!("sucks")
            }
        }

        let name = self.node().raw_node.name.clone();
        self.replicated = Some(Box::new(move |id| format!("{}/aux_identity_{}", name, id)))
    }
}

pub struct Target {
    pb: GraphDef,
    devices: Box<[String]>
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
