use crate::graph::*;
use crate::proto::types::DataType;
use crate::proto::attr_value::{AttrValue, AttrValue_oneof_value};

pub trait Strategy {
    /// make the plan, setting the neccesary fields for nodes and tensors and create the aux nodes on target
    fn plan(&mut self, graph: &mut Graph, target: &mut Target);
}

/// trival strategy that just put everything on CPU0
pub struct NotAtAll;

impl Strategy for NotAtAll {
    fn plan(&mut self, graph: &mut Graph, target: &mut Target) {
        for node in graph.nodes.iter() {
            let replicas = &mut unsafe { &mut *(node as *const Node as *mut Node) }.replicas; // it sucks
            replicas.push((0, node.raw_node.name.clone()));

            for (node_id, index) in node.inputs.iter() {
                let tensor = graph.nodes[*node_id].get_output(*index);
                tensor.aggregated = Some(node.raw_node.name.clone());
            }
        }
    }
}

pub struct Naive;

impl Strategy for Naive {
    fn plan(&mut self, graph: &mut Graph, target: &mut Target) {
        unimplemented!()
    }
}

        // if let Replication::Replicas(replicas) = &self.node().replication {
        //     let replicas = replicas.clone();
        //     let index = self.index;
        //     self.replicated = Some(Box::new(move |id| format!("{}:{}", replicas[id], index)))
        // }

        // match self.method {
        //     ReplicationMethod::cache => self.replicate_cache(),
        //     ReplicationMethod::split => unimplemented!(),
        //     _ => unreachable!()
        // }

    // fn replicate_cache(&mut self) {
    //     let target = self.node().graph().target.as_mut().unwrap();
    //     for (id, device) in target.devices.iter().enumerate() {
    //         let mut identity = NodeDef::new();
    //         identity.name = format!("{}/aux_identity_{}", self.node().raw_node.name, id);
    //         identity.op = "Identity".into();
    //         identity.device = device.into();
    //         let dtype = attr(AttrValue_oneof_value::field_type(DataType::DT_FLOAT)); // TODO: get from raw_node
    //         identity.attr.insert("T".into(), dtype);
    //         if let Replication::Singleton(x) = &self.node().replication {
    //             identity.input.push(x.clone())
    //         } else {
    //             panic!("sucks")
    //         }
    //     }

    //     let name = self.node().raw_node.name.clone();
    //     self.replicated = Some(Box::new(move |id| format!("{}/aux_identity_{}", name, id)))
    // }

    // fn replicate_recursive(&mut self) {
    //     if let Replication::Undefined = self.replication {} else {
    //         return
    //     }

    //     let new_replication;
    //     let target = self.graph().target.as_mut().unwrap();

    //     for (id, _) in &self.inputs {
    //         let mut input = &mut self.graph().nodes[*id];
    //         input.replicate_recursive();
    //     }

    //     // temporary logic
    //     if self.raw_node.op == "VariableV2" { // variables cannot be replicated but its output can be cached
    //         target.pb.node.push(self.raw_node.clone());
    //         new_replication = Replication::Singleton(self.raw_node.name.clone());
    //         let x = &mut self.get_output(0);
    //         x.method = ReplicationMethod::cache;
    //     } else if self.raw_node.op == "Placeholder" { // Placeholder cannot be replicated but its outputs can be splited
    //         target.pb.node.push(self.raw_node.clone());
    //         new_replication = Replication::Singleton(self.raw_node.name.clone());
    //         let x = &mut self.get_output(0);
    //         x.method = ReplicationMethod::split;
    //     } else { // general case, assume pure function, simply duplicate the operator using inputs on the same device
    //         let replicas = (0..target.devices.len()).map(|id| {
    //             let mut x = self.raw_node.clone();

    //             // setup name
    //             write!(&mut x.name, "/replica_{}", id).unwrap();
    //             let name = x.name.clone();

    //             // setup device
    //             x.device = target.devices[id].clone();

    //             // setup inputs
    //             x.input.clear();
    //             for (node_id, index) in &self.inputs {
    //                 let node = &self.graph().nodes[*node_id];
    //                 let tensor_name = node.get_output(*index).get_replicated(id);
    //                 x.input.push(tensor_name);
    //             }

    //             target.pb.node.push(x);
    //             name
    //         }).collect();
    //         new_replication = Replication::Replicas(replicas);
    //     }

    //     self.replication = new_replication;
    // }

fn attr(v: AttrValue_oneof_value) -> AttrValue {
    let mut a = AttrValue::new();
    a.value = Some(v);
    a
}
