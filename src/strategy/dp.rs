use std::convert::TryInto;
use crate::strategy::Strategy;
use crate::graph::Target;
use crate::proto::types::DataType;
use crate::proto::attr_value::{AttrValue, AttrValue_oneof_value};
use crate::proto::node_def::NodeDef;
use crate::proto::tensor::TensorProto;

type Graph = crate::graph::Graph<(), ()>;
type Node = crate::graph::Node<(), ()>;
type Tensor = crate::graph::Tensor<(), ()>;

/// aggressively replicate all nodes for data-parallel and use CPU 0 for reduce
pub struct DataParallelOneForAll;

impl Strategy for DataParallelOneForAll {
    type NEX = ();
    type TEX = ();

    fn plan(&mut self, graph: &mut Graph, target: &mut Target) {
        // first pass: set special logic for sepcial ops
        for node in graph.nodes.iter_mut() {
            match &node.raw_node.op[..] {
                "VariableV2" | "Placeholder" => {
                    node.put_on_device(0);
                }
                "ApplyGradientDescent" => {
                    node.put_on_device(0);
                    let (id, index) = node.inputs[2]; // the gradient
                    node.graph().nodes[id].get_output(index).aggregate_sum(0, target);
                }
                "Assign" | "RandomUniform" => { // TODO: the whole init tree should not be replicated, and be placed alongside the Variable
                    put_on_cpu0(node, target);
                }
                "NoOp" if node.raw_node.name == "GradientDescent" || node.raw_node.name == "init" => {
                    node.replicas.push((0, node.raw_node.name.clone()));
                }
                _ => {
                    replicate_per_device(node, target);
                }
            }
        }

        // second pass: add default logic for remaining ops
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

/// aggressively replicate all nodes for data-parallel and use NCCL for all-reduce
pub struct DataParallelNccl;

impl Strategy for DataParallelNccl {
    type NEX = ();
    type TEX = ();

    fn plan(&mut self, graph: &mut Graph, target: &mut Target) {
        // first pass: set special logic for sepcial ops
        for node in graph.nodes.iter_mut() {
            match &node.raw_node.op[..] {
                "Placeholder" => {
                    put_on_cpu0(node, target);
                    replicate_split(node.get_output(0), target);
                }
                "ApplyGradientDescent" => {
                    replicate_per_device(node, target);
                    let (id, index) = node.inputs[2]; // the gradient
                    all_reduce_sum_nccl(node.graph().nodes[id].get_output(index), target);
                }
                "RandomUniform" => {
                    put_on_cpu0(node, target);
                }
                "NoOp" if node.raw_node.name == "GradientDescent" || node.raw_node.name == "init" => {
                    put_on_cpu0(node, target);
                }
                _ => {
                    replicate_per_device(node, target);
                }
            }
        }

        // second pass: add default logic for remaining ops
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

                // these rules should not exist, just use them before we get replicability inference right
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

/// aggressively replicate all nodes for data-parallel and use naive ring for all reduce
pub struct DataParallelRing;

impl Strategy for DataParallelRing {
    type NEX = ();
    type TEX = ();

    fn plan(&mut self, graph: &mut Graph, target: &mut Target) {
        // first pass: set special logic for sepcial ops
        for node in graph.nodes.iter_mut() {
            match &node.raw_node.op[..] {
                "Placeholder" => {
                    put_on_cpu0(node, target);
                    replicate_split(node.get_output(0), target);
                }
                "ApplyGradientDescent" => {
                    replicate_per_device(node, target);
                    let (id, index) = node.inputs[2]; // the gradient
                    all_reduce_sum_ring_chunked(node.graph().nodes[id].get_output(index), target);
                }
                "RandomUniform" => {
                    put_on_cpu0(node, target);
                }
                "NoOp" if node.raw_node.name == "GradientDescent" || node.raw_node.name == "init" => {
                    put_on_cpu0(node, target);
                }
                _ => {
                    replicate_per_device(node, target);
                }
            }
        }

        // second pass: add default logic for remaining ops
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

                // these rules should not exist, just use them before we get replicability inference right
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
