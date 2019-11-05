use oh_my_rust::*;
use std::convert::TryInto;
use std::rc::Rc;
use std::cell::RefCell;
use crate::strategy::Strategy;
use crate::graph::*;

type Group = Rc<RefCell<Vec<usize>>>;

#[derive(Default, Clone)]
pub struct NEX {
    group: Option<Group>,
    is_descendant_of_input: bool
}

#[derive(Default, Clone)]
pub struct TEX {
    has_batch_dimension: bool
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

    /// 1. mark tensors that has batchsize dimension with hand-crafted whitelist rules
    /// 2. group the nodes so that a.) all nodes inside a group is splittable and b.) all cross-group tensors are splittable
    /// 3. if all nodes in a group are replicated, use split, otherwise all replications are cache.
    #[allow(clippy::cognitive_complexity)]
    fn plan(&mut self, graph: &mut Graph, target: &mut Target) {
        let n = target.devices.len();

        // mark batch splittablity
        for node in graph.nodes.iter_mut() {
            node.extra.is_descendant_of_input = node.inputs.iter().any(|(id, _)| {
                let input = &node.graph().nodes[*id];
                input.extra.is_descendant_of_input || input.raw_node.op == "Placeholder"
            });

            match &node.raw_node.op[..] {
                "Placeholder" | "Conv2D" | "MaxPool" | "MatMul" | "Conv2DBackpropInput" | "BiasAdd" => node.get_output(0).extra.has_batch_dimension = true,
                "Cast" | "ZerosLike" |"GreaterEqual" | "Select" | "Mul" | "Add" | "Sub" | "Neg" | "Log1p" | "Exp" |
                "Squeeze" | "Identity" | "Sigmoid" | "LeakyRelu" | "Relu" | "Tanh" => follow(node, 0, 0),
                _ => {}
                // todo: matmul has an attr that transpose the input on the fly
                // todo: shape -> fill or shape -> broadcast also gives a splittable tensor
            }
        }

        for node in graph.nodes.iter_mut() {
            if node.raw_node.op == "ApplyGradientDescent" {
                let (id, index) = &node.inputs[2];
                let input = &mut node.graph().nodes[*id].get_output(*index);
                input.extra.has_batch_dimension = false;
            }
        }

        // grouping
        for (node_id, node) in graph.nodes.iter_mut().enumerate() {
            if !node.extra.is_descendant_of_input { // if it is not a descendant of input, then it does not belong to any group
                continue
            }

            if node.raw_node.op == "ApplyGradientDescent" { // never in a group
                continue
            }

            for (input_id, index) in node.inputs.iter() {
                let input = &mut node.graph().nodes[*input_id];
                if input.extra.group.is_some() && !input.get_output(*index).extra.has_batch_dimension { // should be attached into the same group
                    let input_group = input.extra.group.as_ref().cloned().unwrap();
                    match &node.extra.group {
                        None => { // this node is not yet assigned into a group, so we just add it into the group of the input
                            node.extra.group = Some(input_group.clone());
                            input_group.borrow_mut().push(node_id);
                        }
                        // this node already belongs to a group that is different from the one of the input. We merge the input group into the current group
                        Some(group) if &**group as *const _ != &*input_group as *const _ => {
                            for i in input_group.borrow().iter() {
                                node.graph().nodes[*i].extra.group = Some(group.clone());
                                group.borrow_mut().push(*i);
                            }
                        }
                        Some(_) => {} // this node already has the same group with the input. Nothing to do here.
                    }
                }
            }

            if node.extra.group.is_none() { // no constraint, assign a new group
                node.extra.group = Some(Rc::new(RefCell::new(vec![node_id])));
            }
        }

        // do replications as the user requested
        for node in graph.nodes.iter_mut() {
            let s = self.strategy_map.get(&node.raw_node.name).copied();

            match &node.raw_node.op[..] {
                // TODO: RandomUniform, NoOp
                "Placeholder" | "NoOp" => node.put_on_device(0), // ignore decision and put on device 0
                "ApplyGradientDescent" | "Assign" => { // ignore decision and put along with the variable
                    let var = &node.graph().nodes[node.inputs[0].0];
                    if var.replicated().unwrap() {
                        node.put_on_all_devices(target);
                    } else {
                        #[allow(mutable_borrow_reservation_conflict)]
                        node.put_on_device(var.replicas[0].0);
                    }
                }
                _ => match s {
                    Some(i) if i < n => node.put_on_device(i),
                    Some(_) | None => node.put_on_all_devices(target),
                }
            }
        }

        // only split if the whole group is replicated. Otherwise go cache (default).
        let mut visited_groups = std::collections::BTreeSet::new();
        for node in graph.nodes.iter_mut() {
            if node.extra.group.is_some() && !visited_groups.contains(&node.extra.group.as_ref().map(|x| &*x.borrow() as *const _).unwrap()) {
                visited_groups.insert(node.extra.group.as_ref().map(|x| &*x.borrow() as *const _).unwrap());
                // info!("{}, {:?}", visited_groups.len(), node.extra.group.as_ref().unwrap().borrow().iter().map(|x| node.graph().nodes[*x].raw_node.name.clone()).collect::<Vec<_>>());
                let group = &node.extra.group.as_ref().unwrap().borrow();
                if group.iter().copied().all(|x| node.graph().nodes[x].replicated().unwrap()) {
                    for member in group.iter() {
                        let member = &mut node.graph().nodes[*member];
                        for (input, reptype) in member.inputs.iter().zip(member.input_replication_types.iter_mut()) {
                            let input = node.graph().nodes[input.0].get_output(input.1);
                            if input.node().extra.is_descendant_of_input && input.extra.has_batch_dimension {
                                *reptype = ReplicationType::Split
                            }
                        }
                    }
                }
            }
        }

        for node in graph.nodes.iter_mut() {
            if node.raw_node.op == "ApplyGradientDescent" {
                let (id, index) = &node.inputs[2];
                assert!(node.extra.group.is_none() || node.extra.group.as_ref().unwrap().borrow().len() == 1); // it either doesn't in a group, or it is its only member
                if node.replicated().unwrap() {
                    let s = self.strategy_map.get(&node.raw_node.name).copied();
                    let value = &mut node.graph().nodes[*id];
                    if value.replicated().unwrap() && value.splitted() {
                        if s == Some(n) { // PS
                            value.get_output(*index).aggregate_sum(node.replicas[0].0, target);
                        } else { // Ring reduce
                            value.get_output(*index).all_reduce_ring(target);
                        }
                    }
                }
            }
        }
    }
}

fn follow(node: &mut Node, input_index: usize, output_index: usize) {
    let (id, index) = node.inputs[input_index];
    node.get_output(output_index).extra.has_batch_dimension = node.graph().nodes[id].get_output(index).extra.has_batch_dimension
}
