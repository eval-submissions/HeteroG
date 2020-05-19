use oh_my_rust::*;
use std::convert::TryInto;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, VecDeque, HashMap};
use std::sync::{Arc, Mutex};
use std::cmp;
use crate::misc::{Target, Profiler};
use crate::graph::Form;
use crate::proto::types::DataType;
use crate::proto::attr_value::{AttrValue, AttrValue_oneof_value};
use crate::proto::node_def::NodeDef;
use crate::proto::tensor::TensorProto;
use crate::simulator::{GRPC_LATENCY, FALLBACK_NCCL_MODEL};

pub fn heft(target: &mut Target, profiler: &impl Profiler) {
    let name_dict: BTreeMap<String, usize> = target.pb.node.iter().enumerate().map(|(i, x)| (x.name.clone(), i)).collect();
    let device_dict: BTreeMap<&String, usize> = target.devices.iter().enumerate().map(|(i, x)| (x, i)).collect();
    let mut ranks = vec![Option::<u64>::None; name_dict.len()];
    let mut succs = vec![BTreeSet::<usize>::new(); name_dict.len()];

    for (node, succ) in target.pb.node.iter().zip(succs.iter_mut()) {
        for input in node.input.iter() {
            let input = if input.starts_with('^') {
                &input[1..]
            } else {
                parse_input(input).0
            };

            succ.insert(name_dict[input]);
        }
    }

    let mut stack: Vec<_> = (0..name_dict.len()).collect();
    while let Some(i) = stack.pop() {
        if ranks[i].is_some() {
            continue
        }

        if succs[i].iter().any(|&j| ranks[j].is_none()) {
            stack.push(i);
            stack.extend(succs[i].iter());
            continue
        }

        let device_id = device_dict[&target.pb.node[i].device];
        let time = succs[i].iter().map(|&j| ranks[j].unwrap()).max().unwrap_or(0) +
                   profiler.profile(&target.pb.node[i], device_id).unwrap_or(0) +
                   1; // additional rank to prevent ties on zero-time op which may cause dead locks
        ranks[i] = Some(time)
    }

    // for (node, rank) in target.pb.node.iter().zip(ranks.iter()) {
    //     info!("{}: {}", node.name, rank.unwrap())
    // }

    let non_dangling_nodes = mark_non_dangling_nodes(target);

    for dev in target.devices.iter() {
        let mut list: Vec<_> = (0..name_dict.len()).filter(|&i| {
            let node = &target.pb.node[i];
            non_dangling_nodes.contains(&node.name) && *dev == node.device
        }).collect();

        list.sort_unstable_by_key(|&x| ranks[x].unwrap());
        for window in list.windows(2) {
            let dep = format!("^{}", target.pb.node[window[0]].name);
            target.pb.node[window[1]].input.push(dep)
            // TODO: don't add unnecessary dependencies by depth-first search
            // TODO: take care of ties?
        }
    }
}

fn parse_input(x: &str) -> (&str, usize) {
    match x.find(':') {
        Some(i) => (&x[..i], x[i+1..].parse().unwrap()),
        None => (x, 0)
    }
}

pub fn mark_non_dangling_nodes(target: &Target) -> std::collections::HashSet<String> {
    let sinks: Vec<_> = target.sinks.iter().map(|x| format!("{}/replica_0", x)).collect();

    // note: don't forget control dependency
    let dict: std::collections::HashMap<_, Vec<_>> = target.pb.node.iter().map(|node| {
        (&node.name[..], node.input.iter().map(|x| {
            if x.starts_with('^') {
                return &x[1..]
            }
            match x.find(':') {
                Some(i) => &x[..i],
                None => x
            }
        }).collect())
    }).collect();
    let mut keep = std::collections::HashSet::new();
    let mut queue: std::collections::VecDeque<_> = sinks.iter().map(|x| &x[..]).collect();

    while let Some(x) = queue.pop_front() {
        if keep.insert(x.to_string()) {
            queue.extend(&dict[x]);
        }
    }

    keep
}
