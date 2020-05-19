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

}
