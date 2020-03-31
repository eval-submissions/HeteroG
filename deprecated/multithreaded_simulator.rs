use taken::*;

#[derive(Debug)]
struct Task2<'a> {
    pub content: TaskType<'a>,
    pub wait_for: Mutex<Vec<usize>>,
    pub notify: Vec<usize>,
    pub in_tensors: Vec<TensorBuf>, // note: in_tensors might be less than wait_for because of control dependencies
    pub out_tensors: Vec<TensorBuf>,
    pub eft: u64
}

impl<'a> Task2<'a> {
    fn create(list: &mut Vec<Task2<'a>>, content: TaskType<'a>, wait_for: &[usize], in_tensors: Vec<TensorBuf>, out_tensors: Vec<TensorBuf>) -> usize {
        let task = Task2 { content, wait_for: Mutex::new(wait_for.to_vec()), in_tensors, out_tensors, notify: vec![], eft: 0 };
        let id = list.len();
        for i in wait_for {
            list[*i].notify.push(id);
        }
        list.push(task);
        id
    }
}

pub struct MultiThreadedSimulator {
    /// the value is a binary sorted array contains replica_number and the time required on each device given replicated by that number
    profile_dict: BTreeMap<String, Vec<(usize, Vec<u64>)>>
}

impl MultiThreadedSimulator {
    pub fn new(profile_dict: BTreeMap<String, Vec<(usize, Vec<u64>)>>) -> Self {
        Self { profile_dict }
    }

    fn profile(&self, node: &NodeDef, device_id: usize) -> Option<u64> {
        let origin_name = node.attr.get("_tge_origin")?.get_s();
        // technically we do not need to extract the form if we use a profiler since it will be reflected by the input size.
        let form = Form::from_code(std::str::from_utf8(node.attr.get("_tge_form")?.get_s()).ok()?);
        let nrep = if form.is_part() {
            form.ndev()
        } else {
            1
        };

        let prof = self.profile_dict.get(&String::from_utf8(origin_name.to_vec()).unwrap())?;
        let time = match prof.binary_search_by_key(&nrep, |x| x.0) {
            Ok(i) => prof[i].1[device_id],
            Err(i) => if i >= prof.len() {
                prof[i - 1].1[device_id]
            } else {
                prof[i].1[device_id]
            }
        };

        Some(time)
    }
}

impl Simulator for MultiThreadedSimulator {
    fn evaluate<W: std::io::Write>(&self, target: Target, mut tracer: Option<&mut W>, max_memory: &mut [u64]) -> u64 {
        task!("evaluating graph of {} nodes...", target.pb.node.len());

        let target = &target;
        if let Some(tracer) = &mut tracer { // initialize tracing
            write!(tracer, "[").unwrap();
        }

        let mut queue: std::collections::VecDeque<_> = target.pb.node.iter().cloned().collect();
        let mut visited = BTreeSet::new();
        let mut nodes = vec![];
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
            nodes.push(node);
        }
        let node_dict: BTreeMap<_, _> = nodes.iter().enumerate().map(|(i, x)| (x.name.clone(), i)).collect();
        let device_dict: BTreeMap<_, _> = target.devices.iter().enumerate().map(|(i, x)| (&x[..], i)).collect();
        let collective_groups = analyze_collective_groups(&target.pb.node, &device_dict, &target.nccls);

        // build tasks
        let mut tasks: Vec<Task2> = vec![];
        let mut task_dict: Vec<usize> = vec![]; // the i-th element is the computation task of the i-th node
        let mut tensorbufs = BTreeMap::<_, (u64, usize, bool)>::new(); // TensorBuf -> (size, ref count, activated)
        for (i, node) in nodes.iter().enumerate() {
            let mut in_tensors = vec![];
            let wait_for: Vec<_> = node.input.iter().map(|input| {
                if input.starts_with('^') {
                    return task_dict[node_dict[&input[1..]]]
                }

                let (name, index) = parse_input(&input);
                let input_id = node_dict[name];
                let from = device_dict[&nodes[input_id].device[..]];
                let to = device_dict[&node.device[..]];
                let size = nodes[input_id].attr.get("_tge_input_sizes").and_then(|x| x.get_list().i.get(index)).copied().unwrap_or(0) as _;

                tensorbufs.entry((input_id, index, from)).and_modify(|x| x.1 += 1).or_insert((size, 1, false));
                tasks[task_dict[input_id]].out_tensors.push((input_id, index, from));

                tensorbufs.entry((input_id, index, to)).and_modify(|x| x.1 += 1).or_insert((size, 1, false));
                in_tensors.push((input_id, index, to));

                // note for memory calculation when from == to: we ignore activation of tensorbuf when it is already activated, and count ref for every transfer, so the calculation is correct.
                Task2::create(&mut tasks, TaskType::Transfer {
                    size, path: &target.paths[from * target.devices.len() + to]
                }, &[task_dict[input_id]], vec![(input_id, index, from)], vec![(input_id, index, to)])
            }).collect();

            let id = if node.op == "CollectiveReduce" {
                let instance_key = node.attr["instance_key"].get_i() as _;
                let group_key = node.attr["group_key"].get_i() as _;
                let input_id = node_dict[parse_input(&node.input[0]).0];
                let size = nodes[input_id].attr.get("_tge_input_sizes").and_then(|x| x.get_list().i.get(0)).copied().unwrap_or(0) as _;
                Task2::create(&mut tasks, TaskType::Collective { instance_key, group_key, size }, &wait_for, in_tensors, vec![])
            } else {
                Task2::create(&mut tasks, TaskType::Computation { id: i, gpu: device_dict[&node.device[..]] }, &wait_for, in_tensors, vec![])
            };
            task_dict.push(id);
        }

        let (ready_queue_sender, ready_queue_receiver) = crossbeam_channel::unbounded();
        for (i, task) in tasks.iter_mut().enumerate() {
            if task.wait_for.get_mut().unwrap().is_empty() {
                ready_queue_sender.send(i).unwrap()
            }
        }

        let mut time = Mutex::new(0);
        let current_memory = Mutex::new(max_memory.to_vec());
        let tensorbufs = Mutex::new(tensorbufs);

        let max_memory = Mutex::new(max_memory);

        let gpu_available_time: Vec<_> = (0..target.devices.len()).map(|_| Mutex::new(0)).collect();
        let link_available_time: Vec<_> = (0..target.links.len()).map(|_| Mutex::new(0)).collect();
        let collective_state = Mutex::new(BTreeMap::<usize, Vec<usize>>::new()); // instance_key => [ready task_id]

        const N_THREADS: usize = 8;

        let count = Mutex::new(tasks.len());
        let mut handles = vec![];
        let fuck = std::time::Instant::now();
        for _thread_id in 0..N_THREADS {
            let ready_queue_sender = ready_queue_sender.clone();
            let ready_queue_receiver = ready_queue_receiver.clone();
            take!(ready_queue_sender, ready_queue_receiver, &count, &time, &tasks, &nodes, &gpu_available_time, &link_available_time, &collective_groups, &collective_state);
            take!(&max_memory, &current_memory, &tensorbufs);
            let handle = scoped::thread::spawn(move || {
                loop {
                    {
                        let mut count = count.lock().unwrap();
                        if *count > 0 {
                            *count -= 1
                        } else {
                            for _ in 0..N_THREADS {
                                ready_queue_sender.send(tasks.len()).unwrap() // to notify others to also quit
                            }
                            return
                        }
                    }

                    let task_id = ready_queue_receiver.recv().unwrap();
                    if task_id >= tasks.len() { // done
                        return
                    }
                    let task = &tasks[task_id];

                    let mut finished_tasks = vec![];
                    match task.content {
                        TaskType::Computation { id: node_id, gpu } => {
                            let eft = cmp::max(*gpu_available_time[gpu].lock().unwrap(), *time.lock().unwrap()) + self.profile(&nodes[node_id], gpu).unwrap_or(0);
                            *(&gpu_available_time[gpu]).lock().unwrap() = eft;
                            finished_tasks.push((task_id, eft));
                        }
                        TaskType::Collective { instance_key, group_key, size } => {
                            let mut collective_state = collective_state.lock().unwrap();
                            let ready_list = collective_state.entry(instance_key).or_default();
                            let group = &collective_groups[&group_key];
                            ready_list.push(task_id);
                            if ready_list.len() == group.devices.len() { // all ready
                                debug!("all ready {}", instance_key);
                                let barrier = group.devices.iter().map(|gpu| *gpu_available_time[*gpu].lock().unwrap()).max().expect("bug");
                                let eft = barrier + nccl_time(size, &collective_groups[&group_key].model);
                                for gpu in group.devices.iter() {
                                    *(&gpu_available_time[*gpu]).lock().unwrap() = eft
                                }
                                for task_id in ready_list {
                                    finished_tasks.push((*task_id, eft))
                                }
                            }
                        }
                        TaskType::Transfer { size, path } => {
                            let est = path.iter().fold(*time.lock().unwrap(), |max, link| cmp::max(max, *link_available_time[*link].lock().unwrap()));
                            let eft = est + if !path.is_empty() {
                                let bandwidth = path.iter().fold(std::u64::MAX, |min, link| cmp::min(min, target.links[*link]));
                                size / bandwidth + GRPC_LATENCY
                            } else {
                                0
                            };

                            for link in path {
                                *(&link_available_time[*link]).lock().unwrap() = eft
                            }
                            finished_tasks.push((task_id, eft));
                        }
                    }

                    for (id, eft) in finished_tasks {
                        // TODO: print tracing information

                        // remove used tensorbufs
                        {
                            let current_memory = &mut *current_memory.lock().unwrap();
                            let tensorbufs = &mut *tensorbufs.lock().unwrap();
                            for in_tensor in &tasks[id].in_tensors {
                                let (size, ref_count, _) = tensorbufs.get_mut(in_tensor).expect("bug in memory tracking: use freed tensor");
                                if *ref_count == 1 { // free
                                    current_memory[in_tensor.2] -= *size;
                                    tensorbufs.remove(in_tensor);
                                } else {
                                    *ref_count -= 1;
                                }
                            }

                            // activate generated tensorbufs
                            let max_memory = &mut *max_memory.lock().unwrap();
                            for out_tensor in &tasks[id].out_tensors {
                                let (size, _, activated) = tensorbufs.get_mut(out_tensor).expect("bug in memory tracking: use freed tensor");
                                if !*activated { // it might already be activated since we allow transfer to the same device
                                    *activated = true;
                                    let gpu = out_tensor.2;
                                    current_memory[gpu] += *size;
                                    max_memory[gpu] = cmp::max(current_memory[gpu], max_memory[gpu]);
                                }
                            }
                        }

                        {
                            let time = &mut *time.lock().unwrap();
                            *time = cmp::max(*time, eft);
                        }

                        for notify in tasks[id].notify.clone() { // TODO: the cloning sucks
                            let list = &mut *(&tasks[notify].wait_for).lock().unwrap();
                            list.retain(|x| *x != id);
                            if list.is_empty() {
                                ready_queue_sender.send(notify).unwrap()
                            }
                        }
                    }
                }
            });
            handles.push(handle)
        }
        for handle in handles {
            handle.join().unwrap()
        }

        warn!("multi {}", fuck.elapsed().as_millis());

        *time.get_mut().unwrap()
    }
}
