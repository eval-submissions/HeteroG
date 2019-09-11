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
                    put_on_cpu0(node, target);
                    replicate_cache(node.get_output(0), target);
                }
                "Placeholder" => {
                    put_on_cpu0(node, target);
                    replicate_split(node.get_output(0), target);
                }
                "ApplyGradientDescent" => {
                    put_on_cpu0(node, target);
                    let (id, index) = node.inputs[2]; // the gradient
                    aggregate_sum(node.graph().nodes[id].get_output(index), target);
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

fn put_on_cpu0(node: &mut Node, target: &mut Target) {
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

fn all_reduce_sum_nccl(tensor: &mut Tensor, target: &mut Target) {
    // to all_sum n tensors (can be on the same devie), one should have n NcclAllReduce nodes with the same shared_name attr
    // each node have only *one* input, and should be on the same device of the input. The output of these nodes will be the same

    assert!(tensor.node().replicated().unwrap());

    let name = tensor.node().raw_node.name.clone();
    let index = tensor.index; // clone here because we will move it later

    for (id, replica) in tensor.node().replicas.iter() {
        let device = &target.devices[*id];
        let mut nccl = NodeDef::new();

        nccl.name = format!("{}/aux_nccl_{}_{}", name, index, id);
        nccl.op = "NcclAllReduce".into();
        nccl.device = device.clone();
        nccl.attr.insert("reduction".into(), attr(AttrValue_oneof_value::s(b"sum".to_vec())));
        nccl.attr.insert("T".into(), get_dtype(&tensor.node().raw_node));
        nccl.attr.insert("num_devices".into(), attr(AttrValue_oneof_value::i(tensor.node().replicas.len().try_into().unwrap())));
        nccl.attr.insert("shared_name".into(), attr(AttrValue_oneof_value::s(tensor.original_name().into_bytes())));
        nccl.input.push(format!("{}:{}", replica, index));

        target.pb.node.push(nccl)
    }

    tensor.replicated = Some(Box::new(move |id| format!("{}/aux_nccl_{}_{}", name, index, id)));
}

/// performing chunked ring all reduce for a list of (device_id, tensor_name), returning the name of summed results on each device
/// basename should be unique!
fn all_reduce_sum_ring_chunked(tensor: &Tensor, list: &[(usize, String)], target: &mut Target) -> Vec<String> {
    let n = list.len();
    let basename = tensor.node().raw_node.name.clone();
    let devices: Vec<_> = list.iter().map(|(id, _)| target.devices[*id].clone()).collect();
    let dtype = get_dtype(&tensor.node().raw_node);

    // 1. chunking
    let mut chunks: Vec<Vec<String>> = (0..n).map(|i| {
        let device = &devices[i];
        let name = &list[i].1;

        let mut dim = NodeDef::new();
        dim.name = format!("{}/ring_{}/aux_split_{}/split_dim", basename, tensor.index, i);
        dim.op = "Const".into();
        dim.device = device.into();
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
        split.name = format!("{}/ring_{}/aux_split_{}", basename, tensor.index, i);
        split.op = "Split".into();
        split.device = device.clone();
        split.input.push(format!("{}/ring_{}/aux_split_{}/split_dim", basename, tensor.index, i));
        split.input.push(name.clone());
        split.attr.insert("T".into(), dtype.clone());
        split.attr.insert("num_split".into(),
            attr(AttrValue_oneof_value::i(n.try_into().unwrap())));
        target.pb.node.push(split);

        (0..n).map(|j| {
            format!("{}/ring_{}/aux_split_{}:{}", basename, tensor.index, i, j)
        }).collect()
    }).collect();

    // 2. n-1 rounds of reducing. the last modified chunks (i+n-2) have the full content
    for round in 0..n-1 {
        // at the r round, the r+i chunk on i node is replaced by the sum of r+i and r+i+1
        for i in 0..n {
            let mut add = NodeDef::new();
            add.name = format!("{}/ring_{}/aux_add_{}_{}", basename, tensor.index, i, round);
            add.op = "Add".into();
            add.device = devices[i].clone();
            add.input.push(chunks[i][(round+i) % n].clone());
            add.input.push(chunks[(i+1) % n][(round+i) % n].clone());
            add.attr.insert("T".into(), dtype.clone());
            chunks[i][(round+i) % n] = add.name.clone();
            target.pb.node.push(add);
        }
    }

    // 3. n-1 rounds of gathering
    for round in 0..n-1 {
        for i in 0..n {
            let mut identity = NodeDef::new();

            identity.name = format!("{}/ring_{}/aux_identity_{}_{}", basename, tensor.index, i, round);
            identity.op = "Identity".into();
            identity.device = devices[i].clone();
            identity.attr.insert("T".into(), dtype.clone());
            identity.input.push(chunks[(i+1) % n][i+round+n-1].clone());
            chunks[i][i+round+n-1] = identity.name.clone();
            target.pb.node.push(identity);
        }
    }

    // 4. concating
    for i in 0..n {

    }

    unimplemented!()
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
