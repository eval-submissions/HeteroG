use std::convert::TryInto;
use std::fmt::Write;
use std::rc::Rc;
use protobuf::Message;
use crate::strategy::Strategy;
use crate::graph::Target;
use crate::proto::types::DataType;
use crate::proto::attr_value::{AttrValue, AttrValue_oneof_value};
use crate::proto::node_def::NodeDef;
use crate::proto::tensor::TensorProto;

trait ID {
    fn id(&self) -> *const Self { self }
}

impl ID for Node {}
impl ID for Tensor {}

#[derive(Default, Clone)]
pub struct NEX {
    batch_splitable: bool, // indicating if this node can be splited by the batch dimension. Currently any descendant of input are considered as splitable
}

#[derive(Default, Clone)]
pub struct TEX {
    users: Vec<*const Node>, // back pointer to children
}

type Graph = crate::graph::Graph<NEX, TEX>;
type Node = crate::graph::Node<NEX, TEX>;
type Tensor = crate::graph::Tensor<NEX, TEX>;

pub struct DynamicProgrammingEarliestFinishTime {
    pub profiler: extern fn(*const u8, u32) -> u64
}

impl Strategy for DynamicProgrammingEarliestFinishTime {
    type NEX = NEX;
    type TEX = TEX;

    fn plan(&mut self, graph: &mut Graph, target: &mut Target) {
        unimplemented!();

        // first pass: make auxiliary markers
        for node in graph.nodes.iter_mut() {
            if node.raw_node.op == "Placeholder" {
                node.extra.batch_splitable = true
            } else {
                node.extra.batch_splitable = node.inputs.iter().any(|(id, _)| {
                    node.graph().nodes[*id].extra.batch_splitable
                })
            }

            for (id, index) in node.inputs.iter() {
                node.graph().nodes[*id].get_output(*index).extra.users.push(node.id())
            }
        }

        // main pass: dynamic programming
        let mut current_states: Vec<Rc<State>> = vec![];
        for i in 0..=graph.nodes.len() {
            let mut next_states: Vec<Rc<State>> = vec![];



            current_states = next_states;
        }


        // third pass: add default logic for remaining ops
        for node in graph.nodes.iter_mut() {
            for (node_id, index) in node.inputs.iter() {
                let tensor = node.graph().nodes[*node_id].get_output(*index);
                if tensor.replicated.is_none() && tensor.node().replicated().unwrap() {
                    let replicas = tensor.node().replicas.clone();
                    let index = tensor.index;
                    tensor.replicated = Some(Box::new(move |id| format!("{}:{}", replicas[id].1, index))) // TODO: find the replica in the device rather than assuming it is the i-th one.
                }
                if tensor.aggregated.is_none() && !tensor.node().replicated().unwrap() {
                    tensor.aggregated = Some(tensor.original_name())
                }

                // these rule should not exist, just use them before we get replicability inference right
                if tensor.replicated.is_none() && !tensor.node().replicated().unwrap() {
                    let name = tensor.original_name();
                    tensor.replicated = Some(Box::new(move |_id| name.clone()))
                }
                if tensor.aggregated.is_none() && tensor.node().replicated().unwrap() {
                    tensor.aggregated = Some(format!("{}:{}", tensor.node().replicas[0].1, tensor.index))
                }
            }
        }
    }
}

type Placement = Vec<bool>;

enum Action {
    MoveTo(*const Tensor, usize),
    Place(*const Node, Placement)
}

// no pipeline since we use eft, so we can skip puting multiple replications on the same device
struct State {
    placement: std::collections::BTreeMap<*const Tensor, Placement>, // the "frontier"
    parent: Option<Rc<State>>
}

impl DynamicProgrammingEarliestFinishTime {
    pub fn new(profiler: extern fn(*const u8, u32) -> u64) -> Self {
        DynamicProgrammingEarliestFinishTime { profiler }
    }

    fn step(&mut self) {

    }

    fn eft(&mut self, node: &Node) -> u64 {
        unimplemented!()
    }



    // TODO: profile a Node, rather than a NodeDef? Stub the input and wait for output (both aggregated and replicated)
    fn profile_computation(&self, node_def: &NodeDef) -> u64 {
        if !HEAVY_OPS.contains(&&node_def.op[..]) {
            return 0
        }
        let mut node_def_raw = vec![];
        node_def.write_to_writer(&mut node_def_raw).unwrap();
        (self.profiler)(node_def_raw.as_ptr(), node_def_raw.len().try_into().unwrap())
    }

    fn profile_transfering(&self, tensor: &Tensor, dest: usize, target: &Target) -> usize {
        0
    }

    fn place_spread(&self, node: &mut Node, finish_time: &[u64], target: &Target) -> (Vec<u64>, u64) {
        let mut updated_time = finish_time.to_vec();
        let n = updated_time.len();

        let profilee_proto = prepare_profilee(node, n);
        for (time, device) in updated_time.iter_mut().zip(target.devices.iter()) {
            let mut profilee = profilee_proto.clone();
            profilee.device = device.clone();
            *time += self.profile_computation(&profilee);
        }

        let eft = updated_time.iter().cloned().fold(0, std::cmp::max);
        (updated_time, eft)
    }

    fn place_single(&self, device_id: usize, node: &mut Node, finish_time: &[u64], target: &Target) -> (Vec<u64>, u64) {
        let mut updated_time = finish_time.to_vec();
        let mut profilee = prepare_profilee(node, 1);
        profilee.device = target.devices[device_id].clone();
        let time = self.profile_computation(&profilee);
        updated_time[device_id] += time;

        (updated_time, time)
    }

    fn place_cluster(&self, device_id: usize, node: &mut Node, finish_time: &[u64], target: &Target) -> (Vec<u64>, u64) {
        let mut updated_time = finish_time.to_vec();
        let mut profilee = prepare_profilee(node, finish_time.len());
        profilee.device = target.devices[device_id].clone();
        let time = self.profile_computation(&profilee);
        updated_time[device_id] += time;

        (updated_time, time)
    }

    // fn place(&self, node: &mut Node, finish_time: &mut Vec<u64>, target: &mut Target) {
    //     let mut best = (Placement::Spread, self.place_spread(node, finish_time, target));

    //     // for single and pipe
    //     for device_id in 0..finish_time.len() {
    //         let (updated_time, eft) = self.place_single(device_id, node, finish_time, target);
    //         if eft < best.1 .1 {
    //             best = (Placement::Single(device_id), (updated_time, eft))
    //         }

    //         let (updated_time, eft) = self.place_cluster(device_id, node, finish_time, target);
    //         if eft < best.1 .1 {
    //             best = (Placement::Cluster(device_id), (updated_time, eft))
    //         }
    //     }

    //     println!("{:?} {}", best.0, node.raw_node.name);
    //     actual_place(node, best.0, target);
    //     finish_time.copy_from_slice(&best.1 .0);
    // }
}

fn prepare_profilee(node: &Node, n: usize) -> NodeDef {
    let mut x = node.raw_node.clone();
    x.name = "profilee".to_string();
    x.input = node.inputs.iter().map(|(id, index)| {
        let parent = &node.graph().nodes[*id];
        let mut shape = parent.get_output(*index).get_shape();
        if n > 1 && parent.replicated().unwrap() && parent.extra.batch_splitable && !shape.is_empty() && shape[0] % n == 0 {
            shape[0] /= n;
        }

        let mut buf = String::new();
        for s in shape {
            write!(&mut buf, "{},", s).unwrap()
        }
        buf
    }).collect();

    x
}

const HEAVY_OPS: &[&str] = &[
    "Conv2D", "MatMul", "Conv2DBackpropFilter", "Conv2DBackpropInput"
];
