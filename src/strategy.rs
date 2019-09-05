use std::convert::TryInto;

use crate::graph::*;
use crate::proto::types::DataType;
use crate::proto::attr_value::{AttrValue, AttrValue_oneof_value};
use crate::proto::node_def::NodeDef;
use crate::proto::tensor::TensorProto;

pub trait Strategy {
    /// make the plan, setting the neccesary fields for nodes and tensors and create the aux nodes on target
    fn plan(&mut self, graph: &mut Graph, target: &mut Target);
}

/// trivial strategy that just put everything on CPU0
pub struct NotAtAll;

impl Strategy for NotAtAll {
    fn plan(&mut self, graph: &mut Graph, target: &mut Target) {
        for node in graph.nodes.iter() {
            let replicas = &mut unsafe { &mut *(node as *const Node as *mut Node) }.replicas; // it sucks
            replicas.push((0, node.raw_node.name.clone()));

            for (node_id, index) in node.inputs.iter() {
                let tensor = graph.nodes[*node_id].get_output(*index);
                tensor.aggregated = Some(format!("{}:{}", tensor.node().raw_node.name.clone(), tensor.index));
            }
        }
    }
}

/// aggressively replicate all nodes for data-parallel and use CPU 0 for reduce
pub struct DataParallelOneForAll;

impl Strategy for DataParallelOneForAll {
    fn plan(&mut self, graph: &mut Graph, target: &mut Target) {
        // first pass: set special logic for sepcial ops
        for node in graph.nodes.iter_mut() {
            match &node.raw_node.op[..] {
                "VariableV2" => {
                    put_on_CPU0(node, target);
                    replicate_cache(node.get_output(0), target);
                }
                "Placeholder" => {
                    put_on_CPU0(node, target);
                    replicate_split(node.get_output(0), target);
                }
                "ApplyGradientDescent" => {
                    put_on_CPU0(node, target);
                    let (id, index) = node.inputs[2]; // the gradient
                    aggregate_sum(node.graph().nodes[id].get_output(index), target);
                }
                "Assign" | "RandomUniform" => { // TODO: the whole init tree should not be replicated, and be placed alongside the Variable
                    put_on_CPU0(node, target);
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

fn put_on_CPU0(node: &mut Node, target: &mut Target) {
    if node.replicated().is_none() {
        node.replicas.push((0, node.raw_node.name.clone()));
    }
}

fn replicate_per_device(node: &mut Node, target: &mut Target) {
    for i in 0..target.devices.len() {
        node.replicas.push((i, format!("{}/replica_{}", node.raw_node.name, i)))
    }
}

fn replicate_split(tensor: &mut Tensor, target: &mut Target) {
    assert!(!tensor.node().replicated().unwrap());

    let name = tensor.node().raw_node.name.clone();
    let index = tensor.index; // clone here because we will move it later

    let mut dim = NodeDef::new();
    dim.name = format!("{}/aux_split_{}/split_dim", name, index);
    dim.op = "Const".into();
    dim.device = target.devices[tensor.node().replicas[0].0].clone();
    dim.attr.insert("dtype".into(),
        attr(AttrValue_oneof_value::field_type(DataType::DT_INT32)));
    let mut value = TensorProto::new();
    let shape = crate::proto::tensor_shape::TensorShapeProto::new();
    value.dtype = DataType::DT_INT32;
    value.tensor_shape = protobuf::SingularPtrField::some(shape);
    value.int_val.push(0);
    dim.attr.insert("value".into(),
        attr(AttrValue_oneof_value::tensor(value)));
    target.pb.node.push(dim);

    let mut split = NodeDef::new();
    split.name = format!("{}/aux_split_{}", name, index);
    split.op = "Split".into();
    split.device = target.devices[tensor.node().replicas[0].0].clone();
    split.input.push(format!("{}/aux_split_{}/split_dim", name, index));
    split.input.push(tensor.original_name());
    split.attr.insert("T".into(), get_dtype(&tensor.node().raw_node));
    split.attr.insert("num_split".into(),
        attr(AttrValue_oneof_value::i(target.devices.len().try_into().unwrap())));
    target.pb.node.push(split);

    tensor.replicated = Some(Box::new(move |id| format!("{}/aux_split_{}:{}", name, index, id)))
}

/// direct identity node: no topology and routing considered
fn replicate_cache(tensor: &mut Tensor, target: &mut Target) {
    assert!(!tensor.node().replicated().unwrap());

    let name = tensor.node().raw_node.name.clone();
    let index = tensor.index; // clone here because we will move it later

    for (id, device) in target.devices.iter().enumerate() {
        let mut identity = NodeDef::new();

        identity.name = format!("{}/aux_identity_{}_{}", name, index, id);
        identity.op = "Identity".into();
        identity.device = device.clone();
        identity.attr.insert("T".into(), get_dtype(&tensor.node().raw_node));
        identity.input.push(tensor.original_name());

        target.pb.node.push(identity)
    }

    tensor.aggregated = Some(format!("{}:{}", name, index));
    tensor.replicated = Some(Box::new(move |id| format!("{}/aux_identity_{}_{}", name, index, id)));
}

fn aggregate_sum(tensor: &mut Tensor, target: &mut Target) {
    assert!(tensor.node().replicated().unwrap());

    let name = tensor.node().raw_node.name.clone();
    let index = tensor.index; // clone here because we will move it later

    let mut addn = NodeDef::new();
    addn.name = format!("{}/aux_sum_{}", name, index);
    addn.op = "AddN".into();
    addn.device = target.devices[tensor.node().replicas[0].0].clone();
    addn.attr.insert("N".into(),
        attr(AttrValue_oneof_value::i(tensor.node().replicas.len().try_into().unwrap())));
    addn.attr.insert("T".into(), get_dtype(&tensor.node().raw_node));
    addn.input = tensor.node().replicas.iter().map(|(_, x)| format!("{}:{}", x, tensor.index)).collect();

    target.pb.node.push(addn);
    tensor.aggregated = Some(format!("{}/aux_sum_{}", name, index));
}

// def get_aggregated_split(self, id): # TODO: what if two ops on different devices want the same tensor?
//     if not hasattr(self, 'merged'):
//         target = self.node.graph.target
//         device = self.node.devices[id]
//         create_raw_node(target, 'Const', device,
//             "{}/{}".format(self.node.raw_node.name, "aux_concat_{}/axis".format(self.index)),
//             dtype = { 'type': DT_INT32 },
//             value = { 'tensor': { 'dtype': DT_INT32, 'tensor_shape': {}, 'int_val': [0] } }
//         )
//         create_raw_node(target, 'ConcatV2', device,
//             "{}/{}".format(self.node.raw_node.name, "aux_concat_{}".format(self.index)),
//             *[ "{}:{}".format(replica.name, self.index) for replica in self.node.replicas ],
//             "{}/{}".format(self.node.raw_node.name, "aux_concat_{}/axis".format(self.index)),
//             N = { 'i': len(self.node.replicas) },
//             T = { 'type': DT_FLOAT },
//             Tidx = { 'type': DT_INT32 }
//         )
//         self.merged = "{}/{}".format(self.node.raw_node.name, "aux_concat_{}".format(self.index))

//     return self.merged

fn attr(v: AttrValue_oneof_value) -> AttrValue {
    let mut a = AttrValue::new();
    a.value = Some(v);
    a
}

fn get_dtype(x: &NodeDef) -> AttrValue {
    x.attr.get("dtype".into()).or(x.attr.get("T".into())).unwrap().clone()
}
