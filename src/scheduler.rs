// evaluate a given graph by simulate a scheduler with profile data

use oh_my_rust::*;
use std::convert::TryInto;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, VecDeque};
use std::cmp;
use crate::graph::Target;
use crate::proto::types::DataType;
use crate::proto::attr_value::{AttrValue, AttrValue_oneof_value};
use crate::proto::node_def::NodeDef;
use crate::proto::tensor::TensorProto;

pub trait Scheduler {
    fn evaluate(&mut self, target: &Target) -> u64;
}

pub struct TensorFlowLikeScheduler {
    n: usize,
    profile_dict: BTreeMap<String, u64>
}

impl TensorFlowLikeScheduler {
    pub fn new(n: usize, profile_dict: BTreeMap<String, u64>) -> Self {
        Self { n, profile_dict }
    }

    fn profile(&self, node: &NodeDef, _device_id: usize) -> Option<u64> {
        let origin_name = node.attr.get("_tge_origin")?.get_s();
        let time = self.profile_dict.get(&String::from_utf8(origin_name.to_vec()).unwrap()).copied();
        // technically we do not need to know whether it is replicated if we use a profiler since it will be reflected by the input size.
        time.map(|x| if node.name.as_bytes() == origin_name { // not replicated
            x
        } else { // replicated
            x / self.n as u64
        })
    }
}

#[derive(Debug)]
enum TaskType<'a> {
    Computation { id: usize, gpu: usize },
    Transfering { size: u64, path: &'a [usize] }
}

#[derive(Debug)]
struct Task<'a> {
    pub content: TaskType<'a>,
    pub wait_for: Vec<usize>,
    pub notify: Vec<usize>,
    pub eft: u64
}

impl<'a> Task<'a> {
    fn create(list: &mut Vec<Task<'a>>, content: TaskType<'a>, wait_for: &[usize]) -> usize {
        let task = Task { content, wait_for: wait_for.to_vec(), notify: vec![], eft: 0 };
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

impl Scheduler for TensorFlowLikeScheduler {
    fn evaluate(&mut self, target: &Target) -> u64 {
        task!("evaluate graph of {} nodes", target.pb.node.len());

        let nodes = &target.pb.node;
        let node_dict: BTreeMap<_, _> = nodes.iter().enumerate().map(|(i, x)| (x.name.clone(), i)).collect();
        let device_dict: BTreeMap<_, _> = target.devices.iter().enumerate().map(|(i, x)| (x.clone(), i)).collect();

        // build tasks
        let mut tasks = vec![];
        let mut task_dict = vec![]; // the i-th element is the computation task of the i-th node
        for (i, node) in nodes.iter().enumerate() {
            let wait_for: Vec<_> = node.input.iter().map(|input| {
                if input.starts_with('^') {
                    return task_dict[node_dict[&input[1..]]]
                }

                let (name, index) = parse_input(&input);
                let input_id = node_dict[name];
                let from = device_dict[&nodes[input_id].device];
                let to = device_dict[&node.device];
                let size = nodes[input_id].attr.get("_tge_input_sizes").and_then(|x| x.get_list().i.get(index)).copied().unwrap_or(0) as _;
                Task::create(&mut tasks, TaskType::Transfering {
                    size, path: &target.paths[from * target.devices.len() + to]
                }, &[task_dict[input_id]])
            }).collect();

            let id = Task::create(&mut tasks, TaskType::Computation { id: i, gpu: device_dict[&node.device] }, &wait_for);
            task_dict[i] = id;
        }

        let mut time = 0;
        let mut ongoing_tasks = BinaryHeap::new();
        let mut ready_list: VecDeque<_> = tasks.iter().enumerate().filter(|(_, task)| task.wait_for.is_empty()).map(|(i, _)| i).collect(); // TODO: find the nodes that actually need to be runned (can lead to the terminating node), or assume the DAG is already pruned.
        let mut gpu_avaliable_time = vec![0; target.devices.len()];
        let mut link_avaliable_time = vec![0; target.links.len()];

        loop {
            // schedule ready tasks. Note the scheduled task may or may not start immediatly depending on the GPU/link queue. There may be other tasks become ready before some tasks schedualed earlier actually start.
            while let Some(task_id) = ready_list.pop_front() {
                let task = &mut tasks[task_id];
                match task.content {
                    TaskType::Computation { id: node_id, gpu } => {
                        let eft = cmp::max(gpu_avaliable_time[gpu], time) + self.profile(&nodes[node_id], gpu).unwrap_or(0);
                        gpu_avaliable_time[gpu] = eft;
                        ongoing_tasks.push(OngoingTask { id: task_id, eft });
                    }
                    TaskType::Transfering { size, path } => {
                        let est = path.iter().fold(time, |max, link| cmp::max(max, link_avaliable_time[*link]));
                        let bandwidth = path.iter().fold(std::u64::MAX, |min, link| cmp::min(min, target.links[*link]));
                        let eft = est + size / bandwidth;

                        for link in path {
                            link_avaliable_time[*link] = eft
                        }
                        ongoing_tasks.push(OngoingTask { id: task_id, eft });
                    }
                }
            }

            // move a time step forward
            if let Some(OngoingTask { id, eft }) = ongoing_tasks.pop() {
                time = eft;
                for notify in &tasks[id].notify.clone() { // TODO: the clone sucks
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

fn parse_input(x: &str) -> (&str, usize) {
    match x.find(':') {
        Some(i) => (&x[..i], x[i+1..].parse().unwrap()),
        None => (x, 0)
    }
}
