use std::convert::TryInto;
use crate::strategy::Strategy;
use crate::graph::Target;
use crate::proto::types::DataType;
use crate::proto::attr_value::{AttrValue, AttrValue_oneof_value};
use crate::proto::node_def::NodeDef;
use crate::proto::tensor::TensorProto;

#[derive(Default, Clone)]
pub struct NEX {
    batch_splitable: bool, // indicating if this node can be splited by the batch dimension. Currently any descendant of input are considered as splitable
}

type Graph = crate::graph::Graph<NEX, ()>;
type Node = crate::graph::Node<NEX, ()>;
type Tensor = crate::graph::Tensor<NEX, ()>;

pub struct Custom {
    strategy_map: std::collections::BTreeMap<String, usize>
}

impl Strategy for Custom {
    type NEX = NEX;
    type TEX = ();

    fn plan(&mut self, graph: &mut Graph, target: &mut Target) {
        let n = target.devices.len();

        // first pass: mark batchiness
        for node in graph.nodes.iter_mut() {
            if node.raw_node.op == "Placeholder" {
                node.extra.batch_splitable = true
            } else {
                node.extra.batch_splitable = node.inputs.iter().any(|(id, _)| {
                    node.graph().nodes[*id].extra.batch_splitable
                })
            }
        }

        // second pass: set replicas (using put_* methods)
        for node in graph.nodes.iter_mut() {
            let s = self.strategy_map.get(&node.raw_node.name).copied();

            match &node.raw_node.op[..] {
                "Placeholder" | "RandomUniform" | "NoOp" => put_on_device(node, 0), // ignore decision and put on device 0
                "ApplyGradientDescent" | "Assign" => { // ignore decision and put along with the variable
                    let id = node.inputs[0].0;
                    if node.graph().nodes[id].replicated().unwrap() {
                        put_on_all_devices(node, target);
                    } else {
                        put_on_device(node, node.graph().nodes[id].replicas[0].0);
                    }
                }
                _ => match s {
                    Some(i) if i < n => {
                        put_on_device(node, i)
                    }
                    Some(_) | None => { // replicate by default
                        put_on_all_devices(node, target)
                    }
                }
            }
        }

        // thrid pass: set replicate and aggregate logic (using replicate_* and aggregate_* methods)
        for node in graph.nodes.iter_mut() {
            for (id, index) in node.inputs.iter() {
                let input = &mut node.graph().nodes[*id];
                let tensor = &mut input.get_output(*index);
                match (node.replicated().unwrap(), input.replicated().unwrap()) {
                    (true, true) => match &node.raw_node.op[..] {
                        "ApplyGradientDescent" => {
                            let s = self.strategy_map.get(&node.raw_node.name).copied();
                        }
                        _ => {
                            let name = node.replicas[*id].1.clone();
                            tensor.replicated = Some(Box::new(move |id| format!("{}:{}", name, index)))
                        }
                    }
                    (true, false) => unimplemented!(),
                    (false, true) => unimplemented!(),
                    (false, false) => unimplemented!(),
                }
            }
        }
    }
}
