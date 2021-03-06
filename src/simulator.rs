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

pub const GRPC_LATENCY: u64 = 12;

#[allow(clippy::unreadable_literal)]
pub const FALLBACK_NCCL_MODEL: [f64; 4] = [0.043420241077615454, 368.2013618677043, 0.27766802543921265, 211.91926070037152];

pub trait Simulator {
    /// `memory` should be already zero-initialized and be at least as long as target.device
    fn evaluate<W: std::io::Write>(&self, profiler: &impl Profiler, target: Target, trace: Option<&mut W>, memory: &mut [u64]) -> u64;
}

#[derive(Debug, Default)]
struct CollectiveGroup {
    devices: Vec<usize>,
    model: [f64; 4]
}

#[derive(Debug)]
enum TaskType<'a> {
    Computation { id: usize, gpu: usize },
    Transfer { size: u64, path: &'a [usize] },
    Collective { instance_key: usize, group_key: usize, size: u64 }
}

#[derive(Debug)]
struct Task<'a> {
    pub content: TaskType<'a>,
    pub wait_for: Vec<usize>,
    pub notify: Vec<usize>,
    pub in_tensors: Vec<TensorBuf>, // note: in_tensors might be less than wait_for because of control dependencies
    pub out_tensors: Vec<TensorBuf>,
    pub eft: u64
}

impl<'a> Task<'a> {
    fn create(list: &mut Vec<Task<'a>>, content: TaskType<'a>, wait_for: &[usize], in_tensors: Vec<TensorBuf>, out_tensors: Vec<TensorBuf>) -> usize {
        let task = Task { content, wait_for: wait_for.to_vec(), in_tensors, out_tensors, notify: vec![], eft: 0 };
        let id = list.len();
        for i in wait_for {
            list[*i].notify.push(id);
        }
        list.push(task);
        id
    }
}

#[derive(Eq, PartialEq)]
struct OngoingTask { id: usize, eft: u64 }

impl Ord for OngoingTask {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        Ord::cmp(&self.eft, &other.eft).reverse()
    }
}

impl PartialOrd for OngoingTask {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

type TensorBuf = (usize, usize, usize); // id, index, gpu

// track all tensors, have two fields: activate (the transfer op that receives it) and deactivate (list of ops that use it)
// implementation: transfer task save activate tensor id, tensors is an array contains sizes and ref counts, computation save deactivate tensor id
// consume memory when the activate op is finished, and deactivate when all deactivate ops are done
// TODO: ensure every tensor being transferred, even if the path is empty

pub struct SimpleSimulator;

impl Simulator for SimpleSimulator {
    fn evaluate<W: std::io::Write>(&self, profiler: &impl Profiler, mut target: Target, mut tracer: Option<&mut W>, max_memory: &mut [u64]) -> u64 {
        task!("evaluating graph of {} nodes...", target.pb.node.len());

        if let Some(tracer) = &mut tracer { // initialize tracing
            write!(tracer, "[").unwrap();
        }

        let nodes = sort_nodes(std::mem::replace(&mut target.pb.node, vec![].into()).into_vec());
        let node_dict: HashMap<_, _> = nodes.iter().enumerate().map(|(i, x)| (&x.name[..], i)).collect();
        let device_dict: BTreeMap<_, _> = target.devices.iter().enumerate().map(|(i, x)| (&x[..], i)).collect();
        let collective_groups = analyze_collective_groups(&nodes, &device_dict, &target.nccls);

        // build tasks
        let mut tasks: Vec<Task> = vec![];
        let mut task_dict: Vec<usize> = vec![]; // the i-th element is the computation task of the i-th node
        let mut tensorbufs = BTreeMap::<_, (u64, usize, bool)>::new(); // TensorBuf -> (size, ref count, activated)
        for (i, node) in nodes.iter().enumerate() {
            let mut in_tensors = vec![];
            let wait_for: Vec<_> = node.input.iter().enumerate().map(|(input_index_of_this_node, input)| {
                if input.starts_with('^') {
                    return task_dict[node_dict[&input[1..]]]
                }

                let (name, index) = parse_input(&input);
                let input_id = node_dict[name];
                let from = device_dict[&nodes[input_id].device[..]];
                let to = device_dict[&node.device[..]];
                let size = node.attr.get("_tge_input_sizes").and_then(|x| x.get_list().i.get(input_index_of_this_node)).copied().unwrap_or(0) as _;

                // info!("{}:{} {}->{} {}", name, index, from, to, size);

                tensorbufs.entry((input_id, index, from)).and_modify(|x| x.1 += 1).or_insert((size, 1, false));
                tasks[task_dict[input_id]].out_tensors.push((input_id, index, from));

                tensorbufs.entry((input_id, index, to)).and_modify(|x| x.1 += 1).or_insert((size, 1, false));
                in_tensors.push((input_id, index, to));

                // note for memory calculation when from == to: we ignore activation of tensorbuf when it is already activated, and count ref for every transfer, so the calculation is correct.
                Task::create(&mut tasks, TaskType::Transfer {
                    size, path: &target.paths[from * target.devices.len() + to]
                }, &[task_dict[input_id]], vec![(input_id, index, from)], vec![(input_id, index, to)])
            }).collect();

            let id = if node.op == "CollectiveReduce" {
                let instance_key = node.attr["instance_key"].get_i() as _;
                let group_key = node.attr["group_key"].get_i() as _;
                let size = node.attr.get("_tge_input_sizes").and_then(|x| x.get_list().i.get(0)).copied().unwrap_or(0) as _;
                Task::create(&mut tasks, TaskType::Collective { instance_key, group_key, size }, &wait_for, in_tensors, vec![])
            } else {
                Task::create(&mut tasks, TaskType::Computation { id: i, gpu: device_dict[&node.device[..]] }, &wait_for, in_tensors, vec![])
            };
            task_dict.push(id);
        }

        let mut time = 0;
        let mut ongoing_tasks = BinaryHeap::new();
        let mut ready_list: VecDeque<_> = tasks.iter().enumerate().filter(|(_, task)| task.wait_for.is_empty()).map(|(i, _)| i).collect();
        let mut gpu_available_time = vec![0; target.devices.len()];
        let mut link_available_time = vec![0; target.links.len()];
        let mut current_memory = max_memory.to_vec();
        let mut collective_state: BTreeMap<usize, Vec<usize>> = BTreeMap::new(); // instance_key => [ready task_id]
        let mut collective_available_time = 0;

        loop {
            // schedule ready tasks. Note the scheduled task may or may not start immediately depending on the GPU/link queue. There may be other tasks become ready before some tasks schedualed earlier actually start.
            while let Some(task_id) = ready_list.pop_front() {
                let task = &tasks[task_id];
                match task.content {
                    TaskType::Computation { id: node_id, gpu } => {
                        debug!("{:?} {:?} {:?} {:?} {:?}", gpu, gpu_available_time[gpu], time, nodes[node_id].name, profiler.profile(&nodes[node_id], gpu).unwrap_or(0));
                        let eft = cmp::max(gpu_available_time[gpu], time) + profiler.profile(&nodes[node_id], gpu).unwrap_or(0);
                        gpu_available_time[gpu] = eft;
                        ongoing_tasks.push(OngoingTask { id: task_id, eft });
                    }
                    TaskType::Collective { instance_key, group_key, size } => {
                        let ready_list = collective_state.entry(instance_key).or_default();
                        let group = &collective_groups[&group_key];
                        ready_list.push(task_id);
                        if ready_list.len() == group.devices.len() { // all ready
                            debug!("all ready {}", instance_key);
                            // let barrier = group.devices.iter().map(|gpu| gpu_available_time[*gpu]).max().expect("bug");
                            // let eft = barrier + nccl_time(size, &collective_groups[&group_key].model);
                            // for gpu in group.devices.iter() {
                            //     gpu_available_time[*gpu] = eft;
                            // }
                            let eft = cmp::max(time, collective_available_time) + nccl_time(size, &collective_groups[&group_key].model);
                            collective_available_time = eft;
                            for task_id in ready_list {
                                ongoing_tasks.push(OngoingTask { id: *task_id, eft })
                            }
                        }
                    }
                    TaskType::Transfer { size, path } => {
                        let est = path.iter().fold(time, |max, link| cmp::max(max, link_available_time[*link]));
                        let eft = est + if !path.is_empty() {
                            let bandwidth = path.iter().fold(std::u64::MAX, |min, link| cmp::min(min, target.links[*link]));
                            size / bandwidth + GRPC_LATENCY
                        } else {
                            0
                        };

                        for link in path {
                            link_available_time[*link] = eft
                        }
                        ongoing_tasks.push(OngoingTask { id: task_id, eft });
                    }
                }
            }

            // move a time step forward
            if let Some(OngoingTask { id, eft }) = ongoing_tasks.pop() {
                // print tracing information
                if let Some(tracer) = &mut tracer {
                    match &tasks[id].content {
                        TaskType::Computation { id: node_id, gpu } => {
                            let duration = profiler.profile(&nodes[*node_id], *gpu).unwrap_or(0);
                            if duration != 0 {
                                writeln!(tracer, "{{ \"name\": \"{}\", \"cat\": \"computation\", \"ph\": \"B\", \"ts\": {}, \"pid\": 0, \"tid\": {} }},", nodes[*node_id].name, eft - duration, gpu).expect("fail to write log");
                                writeln!(tracer, "{{ \"name\": \"{}\", \"cat\": \"computation\", \"ph\": \"E\", \"ts\": {}, \"pid\": 0, \"tid\": {} }},", nodes[*node_id].name, eft, gpu).expect("fail to write log");
                            }
                        }
                        TaskType::Collective { instance_key, group_key, size } => {
                            let duration = nccl_time(*size, &collective_groups[group_key].model);
                            let gpu = tasks[id].in_tensors[0].2; // hack
                            if duration != 0 {
                                writeln!(tracer, "{{ \"name\": \"collective_{}\", \"cat\": \"collective\", \"ph\": \"B\", \"ts\": {}, \"pid\": 0, \"tid\": {} }},", instance_key, eft - duration, gpu).expect("fail to write log");
                                writeln!(tracer, "{{ \"name\": \"collective_{}\", \"cat\": \"collective\", \"ph\": \"E\", \"ts\": {}, \"pid\": 0, \"tid\": {} }},", instance_key, eft, gpu).expect("fail to write log");
                            }
                        }
                        TaskType::Transfer { size, path } => if !path.is_empty() {
                            let bandwidth = path.iter().fold(std::u64::MAX, |min, link| cmp::min(min, target.links[*link]));
                            let duration = size / bandwidth + GRPC_LATENCY;
                            for link in *path {
                                writeln!(tracer, "{{ \"name\": \"transfer{}\", \"cat\": \"transfer\", \"ph\": \"B\", \"ts\": {}, \"pid\": 1, \"tid\": {} }},", id, eft - duration, link).expect("fail to write log");
                                writeln!(tracer, "{{ \"name\": \"transfer{}\", \"cat\": \"transfer\", \"ph\": \"E\", \"ts\": {}, \"pid\": 1, \"tid\": {} }},", id, eft, link).expect("fail to write log");
                            }
                        }
                    }
                };

                // remove used tensorbufs
                for in_tensor in &tasks[id].in_tensors {
                    let tensor_buf = tensorbufs.get_mut(in_tensor);
                    if tensor_buf.is_none() {
                        warn!("bug in memory tracking: use freed tensor {:?}", in_tensor);
                        continue
                    }
                    let (size, ref_count, _) = tensor_buf.unwrap();
                    if *ref_count == 1 { // free
                        current_memory[in_tensor.2] -= *size;
                        debug!("memory of {}:{} {} {} -{} {}", nodes[in_tensor.0].name, in_tensor.1, in_tensor.2, time, *size, current_memory[in_tensor.2]);
                        tensorbufs.remove(in_tensor);
                    } else {
                        *ref_count -= 1;
                    }
                }

                // activate generated tensorbufs
                for out_tensor in &tasks[id].out_tensors {
                    let tensor_buf = tensorbufs.get_mut(out_tensor);
                    if tensor_buf.is_none() {
                        warn!("bug in memory tracking: use freed tensor {:?}", out_tensor);
                        continue
                    }
                    let (size, _, activated) = tensorbufs.get_mut(out_tensor).expect("bug in memory tracking: use freed tensor");
                    if !*activated { // it might already be activated since we allow transfer to the same device
                        *activated = true;
                        let gpu = out_tensor.2;
                        current_memory[gpu] += *size;
                        debug!("memory of {}:{} {} {} +{} {}", nodes[out_tensor.0].name, out_tensor.1, out_tensor.2, time, *size, current_memory[out_tensor.2]);
                        max_memory[gpu] = cmp::max(current_memory[gpu], max_memory[gpu]);
                    }
                }

                time = eft;
                for notify in &tasks[id].notify.clone() { // TODO: the cloning sucks
                    let list = &mut tasks[*notify].wait_for;
                    list.retain(|x| *x != id);
                    if list.is_empty() {
                        ready_list.push_back(*notify)
                    }
                }
            } else { // finally done
                break
            }
        }

        time
    }
}

// use crossbeam_channel;
// include!("../deprecated/multithreaded_simulator.rs");

fn parse_input(x: &str) -> (&str, usize) {
    match x.find(':') {
        Some(i) => (&x[..i], x[i+1..].parse().unwrap()),
        None => (x, 0)
    }
}

fn sort_nodes(x: Vec<NodeDef>) -> Vec<NodeDef> {
    let mut queue: std::collections::VecDeque<_> = x.into();
    let mut visited = BTreeSet::new();
    let mut result = vec![];
    'outer: while let Some(node) = queue.pop_front() {
        for input in node.input.iter() {
            let input = if input.starts_with('^') {
                &input[1..]
            } else {
                parse_input(input).0
            };
            if !visited.contains(input) {
                queue.push_back(node);
                continue 'outer;
            }
        }

        visited.insert(node.name.clone());
        result.push(node);
    }
    result
}

fn analyze_collective_groups(nodes: &[NodeDef], device_dict: &BTreeMap<&str, usize>, nccl_models: &BTreeMap<String, [f64; 4]>) -> BTreeMap<usize, CollectiveGroup> {
    let mut collective_groups: BTreeMap<usize, Vec<&str>> = BTreeMap::new();
    let mut representative_instance: BTreeMap<usize, usize> = BTreeMap::new(); // we use the first instance to represent the group

    let mut tasks: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for device in device_dict.keys() {
        tasks.entry(task_name(device)).or_default().push(device)
    }
    for v in tasks.values_mut() {
        v.sort_unstable()
    }

    for node in nodes.iter() {
        if node.op != "CollectiveReduce" {
            continue
        }

        let group_key = node.attr["group_key"].get_i() as _;
        let instance_key = node.attr["instance_key"].get_i() as _;
        if instance_key != *representative_instance.entry(group_key).or_insert(instance_key) {
            continue
        }

        let group = collective_groups.entry(group_key).or_default();
        group.push(&node.device);
    }

    collective_groups.iter_mut().map(|(&k, v)| {
        let devices = v.iter().map(|&x| device_dict[x]).collect::<Vec<_>>().apply(|x| x.sort_unstable());

        v.sort_unstable();
        let model = if let Some(x) = nccl_models.get(&v.join(",")) {
            *x
        } else {
            let mut set: Vec<_> = v.iter().map(|x| tasks[task_name(x)][0]).collect();
            set.sort_unstable();
            set.dedup();
            if let Some(x) = nccl_models.get(&set.join(",")) {
                *x
            } else {
                warn!("no profiling data for nccl among {}, use a general fallback", set.join(","));
                FALLBACK_NCCL_MODEL
            }
        };

        (k, CollectiveGroup { devices, model })
    }).collect()
}

fn task_name(x: &str) -> &str {
    &x[..x.rfind('/').unwrap()]
}

fn nccl_time(x: u64, nccl_model: &[f64; 4]) -> u64 {
    let t1 = nccl_model[0] * (x >> 10) as f64 + nccl_model[1];
    let t2 = nccl_model[2] * (x >> 10) as f64 + nccl_model[3];
    cmp::max(t1 as _, t2 as _)
}
