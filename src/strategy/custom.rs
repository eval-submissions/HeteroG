use std::convert::TryInto;
use crate::strategy::Strategy;
use crate::graph::*;
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

        // 1. mark bachiness
        for node in graph.nodes.iter_mut() {
            if node.raw_node.op == "Placeholder" {
                node.extra.batch_splitable = true
            } else {
                node.extra.batch_splitable = node.inputs.iter().any(|(id, _)| {
                    node.graph().nodes[*id].extra.batch_splitable
                })
            }
        }

        // 2. do the replication
        for node in graph.nodes.iter_mut() {
            let s = self.strategy_map.get(&node.raw_node.name).copied();

            match &node.raw_node.op[..] {
                "Placeholder" | "RandomUniform" | "NoOp" => node.put_on_device(0), // ignore decision and put on device 0
                "ApplyGradientDescent" | "Assign" => { // ignore decision and put along with the variable
                    let var = &node.graph().nodes[node.inputs[0].0];
                    if var.replicated().unwrap() {
                        node.put_on_all_devices(target);

                        if &node.raw_node.op[..] == "ApplyGradientDescent" { // deal with the gradient
                            let (id, index) = &node.inputs[2];
                            let grad = &mut node.graph().nodes[*id];
                            if grad.replicated().unwrap() && grad.splitted() {
                                if s == Some(n) { // PS
                                    grad.get_output(*index).aggregate_sum(node.replicas[0].0, target);
                                } else { // Ring reduce
                                    grad.get_output(*index).all_reduce_ring(target);
                                }
                            }
                        }
                    } else {
                        #[allow(mutable_borrow_reservation_conflict)]
                        node.put_on_device(var.replicas[0].0);
                    }
                }
                _ => match s {
                    Some(i) if i < n => node.put_on_device(i),
                    Some(_) | None => { // replicate by default
                        node.put_on_all_devices(target);

                        let nodes = &node.graph().nodes; // hack to circumvent the double borrow conflict
                        for (input, reptype) in node.inputs.iter().zip(node.input_replication_types.iter_mut()) {
                            if nodes[input.0].extra.batch_splitable { // split if splittable
                                *reptype = ReplicationType::Split
                            }
                        }
                    }
                }
            }
        }
    }
}
