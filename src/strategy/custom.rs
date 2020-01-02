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

pub struct Custom {
    pub strategy_map: std::collections::BTreeMap<String, (Vec<usize>, bool)> // devices (the same definition of form), is Ring reduce
}

impl Strategy for Custom {
    type NEX = NEX;
    type TEX = TEX;

    /// 1. mark tensors that has batchsize dimension with hand-crafted whitelist rules
    /// 2. group the nodes so that a.) all nodes inside a group is splittable and b.) all cross-group tensors are splittable
    /// 3. if all nodes in a group are replicated, use split, otherwise all replications are cache.
    #[allow(clippy::cognitive_complexity)]
    fn plan(&mut self, graph: &mut Graph, target: &mut Target) {
        // mark batch splittability
        for node in graph.nodes.iter_mut() {
            node.extra.is_descendant_of_input = node.inputs.iter().any(|x| {
                let input = &node.graph().nodes[x.0];
                input.extra.is_descendant_of_input || input.raw_node.op == "Placeholder" || input.raw_node.op == "IteratorGetNext"
            });

            match &node.raw_node.op[..] {
                "Placeholder" | "IteratorGetNext" | "Conv2D" | "MaxPool" | "MatMul" | "Conv2DBackpropInput" | "BiasAdd" => node.get_output(0).extra.has_batch_dimension = true,
                "Cast" | "ZerosLike" |"GreaterEqual" | "Neg" | "Log1p" | "Exp" |
                "Squeeze" | "Identity" | "Sigmoid" | "LeakyRelu" | "Relu" | "Tanh" => follow(node, 0, 0),
                "Add" | "Sub" | "Mul" => any_of(node, &[0, 1], 0),
                _ => {}
                // todo: Select?
                // todo: matmul has an attr that transpose the input on the fly
                // todo: shape -> fill or shape -> broadcast also gives a splittable tensor
            }
        }

        for node in graph.nodes.iter_mut() {
            if node.raw_node.op == "ApplyGradientDescent" {
                let (id, index, _) = &node.inputs[2];
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

            for (input_id, index, _) in node.inputs.iter() {
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
            let s = self.strategy_map.get(&node.raw_node.name).cloned();

            match &node.raw_node.op[..] {
                // TODO: RandomUniform, NoOp
                "Placeholder" | "IteratorGetNext" | "NoOp" => node.put_on_devices(&[0]), // ignore decision and put on device 0
                "ApplyGradientDescent" | "Assign" => { // ignore decision and put along with the variable
                    let var = &node.graph().nodes[node.inputs[0].0];
                    node.put_on_devices(&var.form.devices);
                }
                _ => match s {
                    Some((devices, _)) => node.put_on_devices(&devices),
                    None => node.put_on_devices(&(0..target.ndev()).collect::<Vec<_>>()),
                }
            }
        }

        // only split if the whole group is replicated the same times. Otherwise go cache (default).
        let mut visited_groups = std::collections::BTreeSet::new();
        for node in graph.nodes.iter_mut() {
            if node.extra.group.is_some() && !visited_groups.contains(&node.extra.group.as_ref().map(|x| &*x.borrow() as *const _).unwrap()) {
                visited_groups.insert(node.extra.group.as_ref().map(|x| &*x.borrow() as *const _).unwrap());
                // info!("{}, {:?}", visited_groups.len(), node.extra.group.as_ref().unwrap().borrow().iter().map(|x| node.graph().nodes[*x].raw_node.name.clone()).collect::<Vec<_>>());
                let group = &node.extra.group.as_ref().unwrap().borrow();
                let n = node.form.ndev();
                if n > 1 && group.iter().copied().all(|x| node.graph().nodes[x].form.ndev() == n) {
                    for member in group.iter() {
                        let member = &mut node.graph().nodes[*member];
                        for (id, index, kind) in member.inputs.iter_mut() {
                            let input = node.graph().nodes[*id].get_output(*index);
                            if (input.node().raw_node.op == "Placeholder" || input.node().raw_node.op == "IteratorGetNext" || input.node().extra.is_descendant_of_input) && (group.contains(id) || input.extra.has_batch_dimension) {
                                *kind = FormKind::Part;
                                member.form.kind = FormKind::Part;
                            }
                        }
                    }
                }
            }
        }

        for node in graph.nodes.iter_mut() {
            if node.raw_node.op == "ApplyGradientDescent" {
                let (id, index, _) = &node.inputs[2];
                assert!(node.extra.group.is_none() || node.extra.group.as_ref().unwrap().borrow().len() == 1); // it either doesn't in a group, or it is its only member
                if node.replicated().unwrap() {
                    let s = self.strategy_map.get(&node.raw_node.name).cloned();
                    let grad = &mut node.graph().nodes[*id].get_output(*index);
                    if grad.node().form.is_part() { // is_part implies ndev > 1
                        let full = match s {
                            // alternatively, we can allow this and perform a post transfer?
                            Some((_, true)) if grad.node().form.devices == node.form.devices => grad.all_reduce_ring(&grad.node().form, &node.form.clone(), target),
                            _ => {
                                let x = grad.aggregate_sum(&grad.node().form, &node.form.clone().apply(|x| x.devices.truncate(1)), target);
                                if node.form.ndev() > 1 {
                                    (0..node.form.ndev()).map(|_| x[0].clone()).collect()
                                } else {
                                    x
                                }
                            }
                        };
                        grad.forms.insert(node.form.clone(), full);
                    }
                }
            }
        }
    }
}

fn follow(node: &mut Node, input_index: usize, output_index: usize) {
    let (id, index, _) = node.inputs[input_index];
    node.get_output(output_index).extra.has_batch_dimension = node.graph().nodes[id].get_output(index).extra.has_batch_dimension
}

fn any_of(node: &mut Node, input_indexes: &[usize], output_index: usize) {
    node.get_output(output_index).extra.has_batch_dimension = input_indexes.iter().any(|input_index| {
        let (id, index, _) = node.inputs[*input_index];
        node.graph().nodes[id].get_output(index).extra.has_batch_dimension
    })
}
