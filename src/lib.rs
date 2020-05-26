#![allow(irrefutable_let_patterns)]
#![allow(dead_code, unused_imports)]
#![allow(non_camel_case_types)]
#![deny(bare_trait_objects)]
#![warn(clippy::all)]

use oh_my_rust::*;
use protobuf::{Message, parse_from_bytes};
use simulator::Simulator;
use std::collections::BTreeMap;
use graph::Graph;
use misc::{Target, DataProfiler};

pub mod misc;
pub mod proto;
pub mod graph;
pub mod editor;
pub mod polishing;
pub mod simulator;
pub mod scheduler;

#[no_mangle]
unsafe extern fn create_graph(pb: *const u8, pb_len: u32) -> *mut Graph {
    let pb = std::slice::from_raw_parts(pb, pb_len as usize);
    let g: proto::graph::GraphDef = parse_from_bytes(pb).unwrap();

    Box::leak(Graph::new(&g.node))
}

#[no_mangle]
unsafe extern fn destroy_graph(graph: *mut Graph) {
    free(graph)
}

#[no_mangle]
unsafe extern fn set_option(graph: *mut Graph, name: *const u8, name_len: u32, value: *const u8, value_len: u32) {
    let name = std::str::from_utf8(std::slice::from_raw_parts(name, name_len as usize)).unwrap();
    let value = std::str::from_utf8(std::slice::from_raw_parts(value, value_len as usize)).unwrap();
    (*graph).options.insert(name.to_string(), value.to_string());
}

#[no_mangle]
unsafe extern fn get_groups(graph: *mut Graph, names_raw: *const u8, names_len: *const u8, result: *mut u32) {
    let names = std::str::from_utf8(std::slice::from_raw_parts(names_raw, names_len as usize)).unwrap().split_ascii_whitespace();
    let result = std::slice::from_raw_parts_mut(result, (*graph).nodes.len()); // the actual length could be shorter
    let groups = (*graph).get_groups();
    let mut group_id = BTreeMap::new();
    let mut id = 0;

    for (name, res) in names.zip(result) {
        if let Some(g) = &groups[name] {
            *res = *group_id.entry(g).or_insert_with(|| { id += 1; id - 1 })
        } else {
            *res = id;
            id += 1
        }
    }
}

#[no_mangle]
unsafe extern fn edit_graph(graph: *mut Graph, target: *mut Target, strategy_raw: *const u8, strategy_len: u32) {
    let strategy_str = std::str::from_utf8(std::slice::from_raw_parts(strategy_raw, strategy_len as usize)).unwrap();
    let strategy = strategy_str.lines().map(|line| {
        let line = line.split_ascii_whitespace().collect::<Vec<_>>();
        let name = <&str>::clone(&line[0]);
        let method = line[1].parse::<u8>().unwrap();
        let places = line[2..].iter().map(|x| x.parse().unwrap()).collect(); // assume sorted
        (name, (places, method))
    }).collect();
    editor::edit(&mut *graph, &mut *target, &strategy)
}

#[no_mangle]
unsafe extern fn reset_graph(graph: *mut Graph) {
    editor::reset(&mut *graph)
}

#[no_mangle]
unsafe extern fn create_target(
    devices_raw: *const u8, devices_len: u32,
    links_raw: *const u8, links_len: u32,
    paths_raw: *const u8, paths_len: u32,
    sinks_raw: *const u8, sinks_len: u32,
    nccls_raw: *const u8, nccls_len: u32
) -> *mut Target {
    let links_str = std::str::from_utf8(std::slice::from_raw_parts(links_raw, links_len as usize)).unwrap();
    let links = links_str.split_ascii_whitespace().map(|x| x.parse().unwrap()).collect();

    let paths_str = std::str::from_utf8(std::slice::from_raw_parts(paths_raw, paths_len as usize)).unwrap();
    let paths = paths_str.lines().map(|x| x.split_ascii_whitespace().map(|x| x.parse().unwrap()).collect()).collect();

    let devices_str = std::str::from_utf8(std::slice::from_raw_parts(devices_raw, devices_len as usize)).unwrap();
    let devices = devices_str.split_ascii_whitespace().map(|x| x.to_owned()).collect();

    let sinks_str = std::str::from_utf8(std::slice::from_raw_parts(sinks_raw, sinks_len as usize)).unwrap();
    let sinks = sinks_str.split_ascii_whitespace().map(|x| x.to_string()).collect();

    let nccls_str = std::str::from_utf8(std::slice::from_raw_parts(nccls_raw, nccls_len as usize)).unwrap();
    let nccls = nccls_str.lines().filter(|x| !x.is_empty()).map(|line| {
        let mut m = [0., 0., 0., 0.];
        let line: Vec<_> = line.split_ascii_whitespace().collect();
        for i in 0..4 {
            m[i] = line[i+1].parse().unwrap()
        }
        (line[0].to_string(), m)
    }).collect();

    let target = Target::new(proto::graph::GraphDef::new(), devices, links, paths, sinks, nccls);
    leak(target)
}

#[no_mangle]
unsafe extern fn destroy_target(target: *mut Target) {
    free(target)
}

#[no_mangle]
unsafe extern fn compute_size(target: *mut Target) -> u32 {
    (*target).pb.compute_size()
}

#[no_mangle]
unsafe extern fn read_protobuf(target: *mut Target, dest: *mut u8) {
    let mut ptr = std::slice::from_raw_parts_mut(dest, (*target).pb.get_cached_size() as usize);
    (*target).pb.write_to_writer(&mut ptr).unwrap()
}

#[no_mangle]
unsafe extern fn compile(graph: *mut Graph, target: *mut Target) {
    (*graph).compile(&mut *target)
}

#[no_mangle]
unsafe extern fn create_profiler(profile_data: *const u8, profile_len: u32) -> *mut DataProfiler {
    let profile_str = std::str::from_utf8(std::slice::from_raw_parts(profile_data, profile_len as usize)).unwrap();
    let mut profile_dict: BTreeMap<String, Vec<(usize, Vec<u64>)>> = BTreeMap::new();
    for line in profile_str.lines() {
        let line = line.split_ascii_whitespace().collect::<Vec<_>>();
        let name = line[0].to_string();
        let nrep = line[1].parse().unwrap();
        let times = line[2..].iter().map(|x| x.parse().unwrap()).collect();
        let v = profile_dict.entry(name).or_default();
        let pos = v.binary_search_by_key(&nrep, |x| x.0).unwrap_or_else(|e| e);
        v.insert(pos, (nrep, times))
    };
    leak(DataProfiler { data: profile_dict })
}

#[no_mangle]
unsafe extern fn destroy_profiler(profiler: *mut DataProfiler) {
    free(profiler)
}

#[no_mangle]
unsafe extern fn heft_rank(target: *mut Target, profiler: *const DataProfiler) {
    scheduler::heft_rank(&mut *target, &*profiler, false)
}

#[no_mangle]
unsafe extern fn heft_control(target: *mut Target, profiler: *const DataProfiler) { // this automatically calls rank inside
    scheduler::heft_control(&mut *target, &*profiler)
}

#[no_mangle]
unsafe extern fn evaluate(target: *mut Target, profiler: *const DataProfiler, trace_path: *const u8, trace_len: u32, memory: *mut u64) -> u64 {
    let simulator = simulator::SimpleSimulator;
    let tracer = if trace_len == 0 {
        None
    } else {
        Some(std::str::from_utf8(std::slice::from_raw_parts(trace_path, trace_len as usize)).unwrap())
    };

    simulator.evaluate(&*profiler, *reclaim(target), tracer.map(|x| std::fs::File::create(x).unwrap()).as_mut(), std::slice::from_raw_parts_mut(memory, (*target).devices.len()))
}

#[no_mangle]
unsafe extern fn remove_collocation_hint(target: *mut Target) {
    polishing::remove_collocation_hint(&mut *target)
}

#[no_mangle]
unsafe extern fn remove_shape_hint(target: *mut Target) {
    polishing::remove_shape_hint(&mut *target)
}

#[no_mangle]
unsafe extern fn destruct_names(target: *mut Target) {
    polishing::destruct_names(&mut *target)
}

#[no_mangle]
unsafe extern fn remove_dangling_nodes(target: *mut Target) {
    polishing::remove_dangling_nodes(&mut *target);
}
