use oh_my_rust::*;
use std::convert::TryInto;
use crate::graph::*;
use std::collections::{BTreeSet, BTreeMap};
use crate::misc::Target;

pub fn edit(graph: &mut Graph, target: &mut Target, strategy: &BTreeMap<&str, (Vec<usize>, u8)>) { // devices (the same definition of form), aggregation_method
    let allow_split_input = graph.options.contains_key("replace_placeholder");

    // do replications as the user requested
    for node in graph.nodes.iter_mut() {
        let s = strategy.get(&node.raw_node.name[..]).cloned();

        match &node.raw_node.op[..] {
            // TODO: RandomUniform, NoOp
            "NoOp" => node.put_on_devices(&[0]), // ignore decision and put on device 0
            "Placeholder" | "IteratorGetNext" if !allow_split_input => node.put_on_devices(&[0]),
            "Assign" => { // ignore decision and put along with the variable
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
    let mut visited_groups = BTreeSet::new();
    for node in graph.nodes.iter_mut() {
        if node.group.is_some() && !visited_groups.contains(&node.group.as_ref().map(|x| x.as_ptr() as *const _).unwrap()) {
            visited_groups.insert(node.group.as_ref().map(|x| x.as_ptr() as *const _).unwrap());
            if graph.options.get("log_groups").map(|x| x == "True").unwrap_or(false) {
                 info!("group {}: {:?}", visited_groups.len(), node.group.as_ref().unwrap().borrow().iter().map(|x| node.graph().nodes[*x].raw_node.name.clone()).collect::<Vec<_>>());
            }
            let group = &node.group.as_ref().unwrap().borrow();
            let n = node.form.ndev();
            if n > 1 && group.iter().copied().all(|x| node.graph().nodes[x].form.ndev() == n) {
                for member in group.iter() {
                    let member = &mut node.graph().nodes[*member];
                    if member.inputs.is_empty() && member.is_input() {
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
        match &node.raw_node.op[..] {
            "ApplyGradientDescent" => {
                node.form.kind = FormKind::Full;
                let (id, index, _) = &node.inputs[2];
                if node.replicated().unwrap() {
                    let s = strategy.get(&node.raw_node.name[..]).cloned();
                    let grad = &mut node.graph().nodes[*id].get_output(*index);
                    if grad.node().form.is_part() { // is_part implies ndev > 1
                        let full = match s {
                            // alternatively, we can allow this and perform a post transfer?
                            Some((_, m @ 1..=3)) if grad.node().form.devices == node.form.devices => match m {
                                1 => grad.all_reduce_sum_collective(&grad.node().form, &node.form, target),
                                2 => grad.all_reduce_sum_ring(&grad.node().form, &node.form, target),
                                3 => grad.all_reduce_sum_nccl(&grad.node().form, &node.form, target),
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
            },
            "ScatterSub" => {
                node.form.kind = FormKind::Full;
                let (indices_id, indices_index, _) = &node.inputs[1];
                let (updates_id, updates_index, _) = &node.inputs[2];
                assert!(node.graph().nodes[*indices_id].form == node.graph().nodes[*updates_id].form);
                if node.replicated().unwrap() {
                    let s = strategy.get(&node.raw_node.name[..]).cloned();
                    let indices = &mut node.graph().nodes[*indices_id].get_output(*indices_index);
                    if indices.node().form.is_part() {
                        let full = match s {
                            Some((_, 1)) if indices.node().form.devices == node.form.devices => {
                                indices.all_reduce_cat_collective(&indices.node().form, &node.form.clone(), target)
                            },
                            _ => {
                                let x = indices.aggregate_cat(&indices.node().form, &node.form.clone().apply(|x| x.devices.truncate(1)), target);
                                if node.form.ndev() > 1 {
                                    (0..node.form.ndev()).map(|_| x[0].clone()).collect()
                                } else {
                                    x
                                }
                            }
                        };
                        indices.forms.insert(node.form.clone(), full);
                    }

                    let updates = &mut node.graph().nodes[*updates_id].get_output(*updates_index);
                    if updates.node().form.is_part() {
                        let full = match s {
                            Some((_, 1)) if updates.node().form.devices == node.form.devices => {
                                updates.all_reduce_cat_collective(&updates.node().form, &node.form.clone(), target)
                            },
                            _ => {
                                let x = updates.aggregate_cat(&updates.node().form, &node.form.clone().apply(|x| x.devices.truncate(1)), target);
                                if node.form.ndev() > 1 {
                                    (0..node.form.ndev()).map(|_| x[0].clone()).collect()
                                } else {
                                    x
                                }
                            }
                        };
                        updates.forms.insert(node.form.clone(), full);
                    }
                }
            }
            _ => {}
        }
    }
}

pub fn reset(graph: &mut Graph) {
    for node in graph.nodes.iter_mut() {
        node.form = Form { kind: FormKind::Full, devices: vec![] };
        for (_, _, form) in node.inputs.iter_mut() {
            *form = FormKind::Full
        }
    }
}
