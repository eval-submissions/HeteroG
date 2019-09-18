#![allow(irrefutable_let_patterns)]
#![allow(dead_code, unused_imports)]
#![allow(non_camel_case_types)]
#![deny(bare_trait_objects)]

use oh_my_rust::*;
use protobuf::{Message, parse_from_bytes};
use std::convert::TryInto;

mod proto;
mod graph;
mod strategy;
mod polishing;

type Strategy = Box<dyn strategy::Strategy>;

struct Context(Strategy, Box<graph::Graph>, graph::Target); // NOTE: we keep the graph in box since there are pointers inside nodes that refers to the graph. TODO: use Pin for the box to gurarantee that they are not moved

#[no_mangle]
unsafe extern fn tge(strategy: *mut Strategy, pb: *const u8, pb_len: i32, devices: *const u8, devices_len: i32) -> *mut Context {
    let pb = std::slice::from_raw_parts(pb, pb_len.try_into().unwrap());
    let g: proto::graph::GraphDef = parse_from_bytes(pb).unwrap();
    let graph = graph::Graph::new(g.node.iter());

    let devices_str = std::str::from_utf8(std::slice::from_raw_parts(devices, devices_len.try_into().unwrap())).unwrap();
    let devices: Vec<_> = devices_str.split_ascii_whitespace().map(|x| x.to_owned()).collect();
    let target = graph::Target::new(proto::graph::GraphDef::new(), devices.into_boxed_slice());

    Box::leak(Box::new(Context(*Box::from_raw(strategy), graph, target)))
}

#[no_mangle]
extern fn not_at_all() -> *mut Strategy {
    let strategy = strategy::NotAtAll;
    Box::leak(Box::new(Strategy::from(Box::new(strategy))))
}

#[repr(u8)]
enum CommunicationMethod {
    NONE=0, PS0=1, RING=2, NCCL=3
}

#[no_mangle]
extern fn data_parallel(inner: CommunicationMethod, outer: CommunicationMethod) -> *mut Strategy {
    let strategy = match (inner, outer) {
        (CommunicationMethod::PS0, CommunicationMethod::NONE) => Strategy::from(Box::new(strategy::DataParallelOneForAll)),
        (CommunicationMethod::RING, CommunicationMethod::NONE) => Strategy::from(Box::new(strategy::DataParallelRing)),
        (CommunicationMethod::NCCL, CommunicationMethod::NONE) => Strategy::from(Box::new(strategy::DataParallelNccl)),
        _ => unimplemented!()
    };

    Box::leak(Box::new(strategy))
}

#[no_mangle]
unsafe extern fn compile(ctx: *mut Context) -> u32 {
    let Context(strategy, graph, target) = &mut *ctx;
    strategy.plan(graph, target);
    graph.compile(target);
    polishing::remove_colocation_hint(target);
    // polishing::destructify_names(target);
    target.pb.compute_size()
}

#[no_mangle]
unsafe extern fn read_and_destroy(ctx: *mut Context, dest: *mut u8) {
    let Context(_, _, target) = *Box::from_raw(ctx);
    let mut ptr = std::slice::from_raw_parts_mut(dest, target.pb.get_cached_size() as usize);
    target.pb.write_to_writer(&mut ptr).unwrap();
}

// TODO: better error handling: return NULL or -1 rather than panicing
// TODO: properly exposing the polishing methods: maybe add a bits flag to compile and use keyword arguments on the python side
