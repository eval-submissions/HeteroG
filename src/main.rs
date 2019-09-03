#![allow(irrefutable_let_patterns)]
#![allow(dead_code, unused_imports)]
#![allow(non_camel_case_types)]
#![deny(bare_trait_objects)]

use protobuf::{Message, parse_from_bytes};

mod proto;
mod graph;
mod strategy;
mod polishing;

fn main() {
    let raw_bytes = std::fs::read("g.pb").unwrap();
    let g: proto::graph::GraphDef = parse_from_bytes(&raw_bytes).unwrap();

    let devices = [
        "/job:tge/replica:0/task:0/device:GPU:0",
        "/job:tge/replica:0/task:0/device:GPU:1",
        "/job:tge/replica:0/task:1/device:GPU:0",
        "/job:tge/replica:0/task:1/device:GPU:1",
    ];
    let mut strategy = strategy::DataParallelOneForAll;
    let mut target = graph::Target::new(proto::graph::GraphDef::new(), &devices);
    let mut graph = graph::Graph::new(g.node.iter());

    strategy::Strategy::plan(&mut strategy, &mut graph, &mut target);
    graph.compile(&mut target);

    polishing::remove_colocation_hint(&mut target);

    let mut fout = std::fs::File::create("gout.pb").unwrap();
    target.pb.write_to_writer(&mut fout).unwrap();
}

// TODO: control dependency of GradientDescent and init are broken and should be special cased
//       we can first do not replicate them, then fix the control dependencies specially after the compiling done
