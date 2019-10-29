use oh_my_rust::*;
use std::convert::TryInto;
use std::rc::Rc;
use crate::strategy::Strategy;
use crate::graph::*;

struct Group {
    nodes: Vec<usize>
}

#[derive(Default, Clone)]
pub struct NEX {
    group: Option<Rc<Group>>, // the decision group
}

#[derive(Default, Clone)]
pub struct TEX {
    batch_splittable: bool, // if this tensor can be freely splitted or concatted. Note this is different from the difinition in graph.rs. Basically any tensor can have the `splitted` method as long as the node is replicated and one of the inputs is splitted even if this tensor is marked unsplittable here. The property here is if we can generate a `split` field for tensors whose node is not replicated or repliated as cache.
}

type Graph = crate::graph::Graph<NEX, TEX>;
type Node = crate::graph::Node<NEX, TEX>;
type Tensor = crate::graph::Tensor<NEX, TEX>;

pub struct GroupedCustom {
    pub strategy_map: std::collections::BTreeMap<String, usize>
}

impl Strategy for GroupedCustom {
    type NEX = NEX;
    type TEX = TEX;

    fn plan(&mut self, graph: &mut Graph, target: &mut Target) {
        let n = target.devices.len();

        // mark tensors that has batchsize dimension
        for node in graph.nodes.iter_mut() {
            if node.raw_node.name.starts_with("gradients/") {
                continue
            }

            match &node.raw_node.op[..] {
                "Placeholder" => node.get_output(0).extra.batch_splittable = true,
                "Conv2D" | "MaxPool" | "MatMul" => follow(node, 0, 0),
                "Identity" | "Sigmoid" | "LeakyRelu" | "Relu" | "Tanh" => follow(node, 0, 0),
                "BiasAdd" => follow(node, 0, 0),

                // todo: matmul has an attr that transpose the input on the fly
                // todo: shape -> fill or shape -> broadcast also gives a splittable tensor

                // following are known blacklist
                // Shape:0
                // ShapeN
                // Conv2DBackpropFilter
            }
        }

        // whitelist grouping. If two nodes are connected with a tensor that has no batch size dimension, they will be grouped together
        let groups = vec![];
        for (node_id, node) in graph.nodes.iter_mut().enumerate() {
            for (input_id, index) in node.inputs.iter() {
                let input = &mut node.graph().nodes[*input_id];
                if !input.get_output(*index).extra.batch_splittable { // should be attached into the same group
                    let input_group = input.extra.group.unwrap().clone();
                    match node.extra.group {
                        None => { // this node is not yet assigned into a group, so we just add it into the group of the input
                            node.extra.group = Some(input_group);
                            input_group.nodes.push(node_id);
                        }
                        // this node already belongs to a group that is different from the one of the input. We merge the input group into the current group
                        Some(group) if &*group as *const _ != &*input_group as *const _ => {
                            for i in input_group.nodes.iter() {
                                node.graph().nodes[*i].extra.group = Some(group.clone());
                                group.nodes.push(*i);
                            }
                        }
                        Some(group) => {} // this node already has the same group with the input. Nothing to do here.
                    }
                }
            }

            if node.extra.group.is_none() { // no constraint, assign a new group
                node.extra.group = Some(Rc::new(Group { nodes: vec![node_id] }));
            }
        }


        for node in graph.nodes.iter_mut() {
            let mut s = self.strategy_map.get(&node.raw_node.name).copied();

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

fn follow(node: &mut Node, input_index: usize, output_index: usize) {
    let (id, index) = node.inputs[input_index];
    node.graph().nodes[id].get_output(index)
}
