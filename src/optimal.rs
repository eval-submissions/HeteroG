// find the optimal strategy using branch and bound

use oh_my_rust::*;
use std::convert::TryInto;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, VecDeque};
use std::cmp;
use crate::graph::{Graph, Target, Form};
use crate::proto::types::DataType;
use crate::proto::attr_value::{AttrValue, AttrValue_oneof_value};
use crate::proto::node_def::NodeDef;
use crate::proto::tensor::TensorProto;
use std::sync::{Arc};

pub fn find_optimal(nodes: &[NodeDef], target: Target, scheduler: impl crate::scheduler::Scheduler, profile_dict: BTreeMap<String, Vec<u64>>) -> u64 {
    let target = Arc::new(target);
    let scheduler = Arc::new(scheduler);
    let m = target.devices.len();

    // the i-th element is the lower bound of time required to calculate from i-th (included) node to the end.
    let infimum = target.pb.node.iter().map(|node_def| {
        (0..m).map(|i| Some(1. / scheduler.profile(node_def, i)? as f64)).collect::<Option<Vec<_>>>().map(|x| (1.0 / x.iter().sum::<f64>()) as u64).unwrap_or(0)
    }).rev().scan(0, |acc, x| {
        *acc += x;
        Some(*acc)
    }).collect::<Vec<_>>().apply(|x| x.reverse());

    let best = std::u64::MAX;

    0
}
