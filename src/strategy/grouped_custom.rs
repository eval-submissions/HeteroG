use oh_my_rust::*;
use std::convert::TryInto;
use std::rc::Rc;
use std::cell::RefCell;
use crate::strategy::Strategy;
use crate::graph::*;

struct Group {
    nodes: Vec<usize>
}

#[derive(Default, Clone)]
pub struct NEX {
    group: Option<Rc<RefCell<Group>>>, // the decision group. The nodes inside a group can either all be cached (but can be replicated or not) or all be splitted AND replicated
    batch_splittable: bool, // if any of its inputs are Placeholder or batch_splittable
}

#[derive(Default, Clone)]
pub struct TEX {
    batch_splittable: bool, // if this tensor can be freely splitted or concatted. e.g. if it has .cache when the node is replicated as split, or it has .split when the node is not replicated / replicated as cache
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
            if node.raw_node.name.starts_with("gradients/") {
                continue
            }

            node.extra.batch_splittable = node.raw_node.op == "Placeholder" || node.inputs.iter().any(|(id, _)| {
                node.graph().nodes[*id].extra.batch_splittable
            });

            match &node.raw_node.op[..] {
                "Placeholder" => node.get_output(0).extra.batch_splittable = true,
                "Conv2D" | "MaxPool" | "MatMul" => follow(node, 0, 0),
                "Identity" | "Sigmoid" | "LeakyRelu" | "Relu" | "Tanh" => follow(node, 0, 0),
                "BiasAdd" => follow(node, 0, 0),
                _ => {}
                // todo: matmul has an attr that transpose the input on the fly
                // todo: shape -> fill or shape -> broadcast also gives a splittable tensor
            }
        }

        // grouping
        for (node_id, node) in graph.nodes.iter_mut().enumerate() {
            if !node.extra.batch_splittable { // if it is not splittable, then it does not belong to any group
                continue
            }

            for (input_id, index) in node.inputs.iter() {
                let input = &mut node.graph().nodes[*input_id];
                if input.extra.group.is_some() && !input.get_output(*index).extra.batch_splittable { // should be attached into the same group
                    let input_group = input.extra.group.as_ref().unwrap();
                    match &node.extra.group {
                        None => { // this node is not yet assigned into a group, so we just add it into the group of the input
                            node.extra.group = Some(input_group.clone());
                            input_group.borrow_mut().nodes.push(node_id);
                        }
                        // this node already belongs to a group that is different from the one of the input. We merge the input group into the current group
                        Some(group) if &**group as *const _ != &**input_group as *const _ => {
                            for i in input_group.borrow_mut().nodes.iter() {
                                node.graph().nodes[*i].extra.group = Some(group.clone());
                                group.borrow_mut().nodes.push(*i);
                            }
                        }
                        Some(_) => {} // this node already has the same group with the input. Nothing to do here.
                    }
                }
            }

            if node.extra.group.is_none() { // no constraint, assign a new group
                node.extra.group = Some(Rc::new(RefCell::new(Group { nodes: vec![node_id] })));
            }
        }

        // do replications as the user requested
        for node in graph.nodes.iter_mut() {
            let s = self.strategy_map.get(&node.raw_node.name).copied();

            match &node.raw_node.op[..] {
                "Placeholder" | "NoOp" | "RandomUniform" => node.put_on_device(0), // ignore decision and put on device 0
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
            if node.extra.group.is_some() && visited_groups.contains(&node.extra.group.as_ref().map(|x| &*x.borrow() as *const _).unwrap()) {
                visited_groups.insert(node.extra.group.as_ref().map(|x| &*x.borrow() as *const _).unwrap());
                let group = &mut node.extra.group.as_ref().unwrap().borrow_mut().nodes;
                if group.iter().copied().all(|x| node.graph().nodes[x].replicated().unwrap()) {
                    for member in group.iter_mut() {
                        let member = &mut node.graph().nodes[*member];
                        for (input, reptype) in member.inputs.iter().zip(member.input_replication_types.iter_mut()) {
                            if node.graph().nodes[input.0].extra.batch_splittable {
                                *reptype = ReplicationType::Split
                            }
                        }
                    }
                }
            }

            let s = self.strategy_map.get(&node.raw_node.name).copied();

            // deal with special ops
            match &node.raw_node.op[..] {
                "ApplyGradientDescent" | "Assign" => {
                    assert!(node.extra.group.is_none() || node.extra.group.as_ref().unwrap().borrow().nodes.len() == 1); // it either doesn't in a group, or it is its only member

                    // reset rep type if they are set since the node is in a (standalone) group
                    for reptype in node.input_replication_types.iter_mut() {
                        *reptype = ReplicationType::Cache;
                    }

                    if node.replicated().unwrap() {
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
                    }
                }
                _ => {}
            }
        }
    }
}

fn follow(node: &mut Node, input_index: usize, output_index: usize) {
    let (id, index) = node.inputs[input_index];
    node.get_output(output_index).extra.batch_splittable = node.graph().nodes[id].get_output(index).extra.batch_splittable
}
