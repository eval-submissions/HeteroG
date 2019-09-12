#![allow(irrefutable_let_patterns)]
#![allow(dead_code, unused_imports)]
#![allow(non_camel_case_types)]
#![deny(bare_trait_objects)]

use protobuf::{Message, parse_from_bytes};
use oh_my_rust::*;

mod proto;
mod graph;
mod strategy;
mod polishing;

fn main() {
    let raw_bytes = std::io::stdin().read_to_end_alloc().unwrap();
    let g: proto::graph::GraphDef = parse_from_bytes(&raw_bytes).unwrap();

    let devices: Vec<_> = std::env::args().skip(1).collect();
    let mut strategy = strategy::DataParallelRing;
    let mut target = graph::Target::new(proto::graph::GraphDef::new(), devices.into_boxed_slice());
    let mut graph = graph::Graph::new(g.node.iter());

    strategy::Strategy::plan(&mut strategy, &mut graph, &mut target);
    graph.compile(&mut target);

    polishing::remove_colocation_hint(&mut target);

    target.pb.write_to_writer(&mut std::io::stdout()).unwrap();
}

// TODO list
// 2. try topology-aware reduce and broadcasting under the assumption that devices with the same task_id has neglectable communication cost
// 4. currently it panics if only one device is passed. Not a big problem though
