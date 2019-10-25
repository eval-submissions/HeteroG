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
    pub strategy_map: std::collections::BTreeMap<String, usize>
}

impl Strategy for Custom {
    type NEX = NEX;
    type TEX = ();

    fn plan(&mut self, graph: &mut Graph, target: &mut Target) {
        let n = target.devices.len();

        for node in graph.nodes.iter_mut() {
            node.extra.batch_splitable = if node.raw_node.op == "Placeholder" {
                true
            } else if node.raw_node.name.starts_with("gradients/") {
                false
            } else {
                node.inputs.iter().any(|(id, _)| {
                    node.graph().nodes[*id].extra.batch_splitable
                })
            }
        }

        // Another special hack: the two argument of BroadcastGradientArgs must agree
        for node in graph.nodes.iter_mut() {
            if &node.raw_node.op[..] == "BroadcastGradientArgs" {
                let n1 = &node.graph().nodes[node.inputs[0].0];
                let n2 = &node.graph().nodes[node.inputs[1].0];
                if n1.extra.batch_splitable == n2.extra.batch_splitable {
                    *self.strategy_map.entry(n1.raw_node.name.clone()).or_default() = self.strategy_map.get(&n2.raw_node.name).copied().unwrap_or_default()
                } else {
                    self.strategy_map.insert(n1.raw_node.name.clone(), 0);
                    self.strategy_map.insert(n2.raw_node.name.clone(), 0);
                }
            }
        }

        for node in graph.nodes.iter_mut() {
            let mut s = self.strategy_map.get(&node.raw_node.name).copied();

            // special hack for nodes that has unbatched input
            if node.inputs.iter().any(|(id, _)| unbatched(&node.graph().nodes[*id])) {
                let (id, _) = node.inputs.iter().find(|(id, _)| unbatched(&node.graph().nodes[*id])).unwrap();
                let shape = &node.graph().nodes[*id];
                if shape.replicated().unwrap() {
                    s = Some(n);
                } else {
                    s = Some(shape.replicas[0].0)
                }
            }

            match &node.raw_node.op[..] {
                "Placeholder" | "NoOp" => node.put_on_device(0), // ignore decision and put on device 0
                "RandomUniform" => { // if the shape is splitted, split. Otherwise, don't replicate
                    let shape = &node.graph().nodes[node.inputs[0].0];
                    if shape.replicated().unwrap() && shape.splitted() {
                        node.put_on_all_devices(target);
                        node.input_replication_types[0] = ReplicationType::Split;
                    } else {
                        node.put_on_device(0);
                    }
                }
                "ApplyGradientDescent" | "Assign" => { // ignore decision and put along with the variable
                    let var = &node.graph().nodes[node.inputs[0].0];
                    if var.replicated().unwrap() {
                        node.put_on_all_devices(target);

                        // deal with the value
                        let (id, index) = match &node.raw_node.op[..] {
                            "ApplyGradientDescent" => &node.inputs[2],
                            "Assign" => &node.inputs[1],
                            _ => unreachable!()
                        };
                        let value = &mut node.graph().nodes[*id];
                        if value.replicated().unwrap() && value.splitted() {
                            if s == Some(n) { // PS
                                value.get_output(*index).aggregate_sum(node.replicas[0].0, target);
                            } else { // Ring reduce
                                value.get_output(*index).all_reduce_ring(target);
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

fn unbatched(x: &Node) -> bool {
    x.raw_node.op == "Shape" || x.raw_node.op == "ShapeN" || x.raw_node.op == "Conv2DBackpropFilter"
}
