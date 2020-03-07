use oh_my_rust::*;
use std::convert::TryInto;
use std::rc::Rc;
use std::cell::RefCell;
use crate::strategy::Strategy;
use crate::graph::*;

type Group = Rc<RefCell<Vec<usize>>>;

#[derive(Default, Clone)]
pub struct NEX {
}

#[derive(Default, Clone)]
pub struct TEX {
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

    fn plan(&mut self, graph: &mut Graph, target: &mut Target) {
        let allow_split_input = graph.options.contains_key("replace_placeholder");

        // do replications as the user requested
        for node in graph.nodes.iter_mut() {
            let s = self.strategy_map.get(&node.raw_node.name).cloned();

            match &node.raw_node.op[..] {
                // TODO: RandomUniform, NoOp
                "NoOp" => node.put_on_devices(&[0]), // ignore decision and put on device 0
                "Placeholder" | "IteratorGetNext" if !allow_split_input => node.put_on_devices(&[0]),
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
            if node.group.is_some() && !visited_groups.contains(&node.group.as_ref().map(|x| &*x.borrow() as *const _).unwrap()) {
                visited_groups.insert(node.group.as_ref().map(|x| &*x.borrow() as *const _).unwrap());
                if graph.options.get("log_groups").map(|x| x == "True").unwrap_or(false) {
                     info!("group {}: {:?}", visited_groups.len(), node.group.as_ref().unwrap().borrow().iter().map(|x| node.graph().nodes[*x].raw_node.name.clone()).collect::<Vec<_>>());
                }
                let group = &node.group.as_ref().unwrap().borrow();
                let n = node.form.ndev();
                if n > 1 && group.iter().copied().all(|x| node.graph().nodes[x].form.ndev() == n) {
                    for member in group.iter() {
                        let member = &mut node.graph().nodes[*member];
                        if member.inputs.is_empty() && member.is_input() { // work around. Need to sort this out later.
                            member.form.kind = FormKind::Part;
                            continue
                        }
                        for (id, index, kind) in member.inputs.iter_mut() {
                            let input = node.graph().nodes[*id].get_output(*index);
                            if input.has_flag(Tensor::IS_FROM_INPUT) && (group.contains(id) || input.has_flag(Tensor::IS_BATCHED)) {
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
                assert!(node.group.is_none() || node.group.as_ref().unwrap().borrow().len() == 1); // it either doesn't in a group, or it is its only member
                if node.replicated().unwrap() {
                    let s = self.strategy_map.get(&node.raw_node.name).cloned();
                    let grad = &mut node.graph().nodes[*id].get_output(*index);
                    if grad.node().form.is_part() { // is_part implies ndev > 1
                        let full = match s {
                            // alternatively, we can allow this and perform a post transfer?
                            Some((_, true)) if grad.node().form.devices == node.form.devices => match graph.options.get("allreduce_implementation").map(|x| &x[..]) {
                                Some("nccl") => grad.all_reduce_nccl(&grad.node().form, &node.form.clone(), target),
                                Some("collective") => grad.all_reduce_collective(&grad.node().form, &node.form.clone(), target),
                                None => grad.all_reduce_ring(&grad.node().form, &node.form.clone(), target),
                                _ => unreachable!()
                            },
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
