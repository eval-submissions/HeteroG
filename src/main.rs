#![allow(irrefutable_let_patterns)]
#![allow(dead_code, unused_imports)]
#![allow(non_camel_case_types)]
#![deny(bare_trait_objects)]

use protobuf::{Message, parse_from_bytes};

mod proto;
mod graph;
mod strategy;

fn main() {
    let raw_bytes = std::fs::read("g.pb").unwrap();
    let mut g: proto::graph::GraphDef = parse_from_bytes(&raw_bytes).unwrap();

    // for node in g.node.iter_mut() {
    //     node.device = "/device:CPU:0".into()
    // }

    let mut strategy = strategy::NotAtAll;
    let mut target = graph::Target::new(proto::graph::GraphDef::new(), &["/device:CPU:0"]);
    let mut graph: graph::Graph = g.node.iter().collect();

    strategy::Strategy::plan(&mut strategy, &mut graph, &mut target);
    graph.compile(&mut target);

    let mut fout = std::fs::File::create("gout.pb").unwrap();
    g.write_to_writer(&mut fout).unwrap();
}
