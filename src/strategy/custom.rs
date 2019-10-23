use std::convert::TryInto;
use crate::strategy::Strategy;
use crate::graph::Target;
use crate::proto::types::DataType;
use crate::proto::attr_value::{AttrValue, AttrValue_oneof_value};
use crate::proto::node_def::NodeDef;
use crate::proto::tensor::TensorProto;

#[derive(Default, Clone)]
pub struct NEX {
    batch_splitable: bool, // indicating if this node can be splited by the batch dimension. Currently any descendant of input are considered as splitable
}

pub struct TEX {
    aggregate_aux_need_compile: bool,
}

type Graph = crate::graph::Graph<NEX, ()>;
type Node = crate::graph::Node<NEX, ()>;
type Tensor = crate::graph::Tensor<NEX, ()>;

pub struct Custom {
    strategy_map: std::collections::BTreeMap<String, usize>
}

impl Strategy for Custom {
    type NEX = NEX;
    type TEX = ();

    fn plan(&mut self, graph: &mut Graph, target: &mut Target) {
        // first pass: mark batchiness
        for node in graph.nodes.iter_mut() {
            if node.raw_node.op == "Placeholder" {
                node.extra.batch_splitable = true
            } else {
                node.extra.batch_splitable = node.inputs.iter().any(|(id, _)| {
                    node.graph().nodes[*id].extra.batch_splitable
                })
            }
        }

        // second pass: decide whether to replicate and the replication method for special operators
        for node in graph.nodes.iter_mut() {
            let s = self.strategy_map.get(&node.raw_node.name).copied();
            let n = target.devices.len();

            match &node.raw_node.op[..] {
                "VariableV2" => {
                    if s.unwrap() >= n {
                        replicate_per_device(node, target);
                    } else {
                        put_on_device(node, 0);
                    }
                }
                "Placeholder" => { // ignore decision and put on device 0
                    put_on_device(node, 0);
                    replicate_split(node.get_output(0), target);
                }
                "ApplyGradientDescent" => { // ignore decision and put along with variable
                    let (vid, vindex) = node.inputs[0]; // the variable
                    let (gid, gindex) = node.inputs[2]; // the gradient
                    if node.graph().nodes[vid].replicated().unwrap() {
                        // TODO: check is it ps or all-reduce
                        replicate_per_device(node, target);
                        all_reduce_sum_ring_chunked(node.graph().nodes[gid].get_output(gindex), target);
                    } else {
                        all_reduce_sum_ring_chunked(node.graph().nodes[gid].get_output(gindex), target);
                    }
                    // put_on_device(node, 0);
                    // aggregate_sum(node.graph().nodes[id].get_output(index), target);
                }
                "Assign" => { // ignore decision and put along with variable
                    let (id, index) = node.inputs[0];
                    put_on_device(node, 0);
                }
                "RandomUniform" => { // This one just can't be replicated
                    put_on_device(node, 0);
                }
                "NoOp" /*if node.raw_node.name == "GradientDescent" || node.raw_node.name == "init"*/ => {
                    put_on_device(node, 0);
                }
                _ => {
                    match s {
                        Some(i) if i < n => put_on_device(node, i),
                        Some(_) => replicate_per_device(node, target),
                        None => replicate_per_device(node, target) // replicate by default so initialize-related ops won't complaint
                    }
                }
            }
        }

        // thrid pass: implement replicate and aggregate logic
        // TODO: not all sorted. For example, it always insert aggregate_cat even is not used.
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

                if tensor.replicated.is_none() && !tensor.node().replicated().unwrap() {
                    if tensor.node().extra.batch_splitable {
                        replicate_split(tensor, target);
                    } else {
                        replicate_cache(tensor, target);
                    }

                    let name = tensor.original_name();
                    tensor.replicated = Some(Box::new(move |_id| name.clone()))
                }
                if tensor.aggregated.is_none() && tensor.node().replicated().unwrap() {
                    if tensor.node().extra.batch_splitable {
                        aggregate_cat(tensor, target);
                    } else {
                        // TODO: this should never happen!
                        tensor.aggregated = Some(format!("{}:{}", tensor.node().replicas[0].1, tensor.index))
                    }
                }
            }
        }
    }
}

fn put_on_device(node: &mut Node, device_id: usize) {
    if node.replicated().is_none() {
        node.replicas.push((device_id, node.raw_node.name.clone()));
    } else {
        panic!("sucks")
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

    tensor.aggregated = Some(tensor.original_name());
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

fn aggregate_cat(tensor: &mut Tensor, target: &mut Target) {
    assert!(tensor.node().replicated().unwrap());

    let name = tensor.node().raw_node.name.clone();
    let index = tensor.index;

    let mut axis = NodeDef::new();
    axis.name = format!("{}/aux_concat_{}/axis", name, index);
    axis.op = "Const".into();
    axis.device = target.devices[tensor.node().replicas[0].0].clone();
    axis.attr.insert("dtype".into(), attr(AttrValue_oneof_value::field_type(DataType::DT_INT32)));
    let mut value = TensorProto::new();
    let shape = crate::proto::tensor_shape::TensorShapeProto::new();
    value.dtype = DataType::DT_INT32;
    value.tensor_shape = protobuf::SingularPtrField::some(shape);
    value.int_val.push(0);
    axis.attr.insert("value".into(), attr(AttrValue_oneof_value::tensor(value)));
    target.pb.node.push(axis);

    let mut concat = NodeDef::new();
    concat.name = format!("{}/aux_concat_{}", name, index);
    concat.op = "ConcatV2".into();
    concat.device = target.devices[tensor.node().replicas[0].0].clone();
    concat.input = tensor.node().replicas.iter().map(|(_, x)| format!("{}:{}", x, index)).collect();
    concat.input.push(format!("{}/aux_concat_{}/axis", name, index));
    concat.attr.insert("N".into(), attr(AttrValue_oneof_value::i(tensor.node().replicas.len().try_into().unwrap())));
    concat.attr.insert("T".into(), get_dtype(&tensor.node().raw_node));
    concat.attr.insert("Tidx".into(), attr(AttrValue_oneof_value::field_type(DataType::DT_INT32)));

    target.pb.node.push(concat);
    tensor.aggregated = Some(format!("{}/aux_concat_{}", name, index))
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
fn _all_reduce_sum_ring_chunked(tensor: &Tensor, list: &[(usize, String)], target: &mut Target) -> Vec<String> {
    let n = list.len();
    let basename = tensor.node().raw_node.name.clone();
    let devices: Vec<_> = list.iter().map(|(id, _)| target.devices[*id].clone()).collect();
    let dtype = get_dtype(&tensor.node().raw_node);

    // 1. recording the shape
    let shapes: Vec<_> = (0..n).map(|i| {
        let mut shape = NodeDef::new();
        shape.name = format!("{}/ring_{}/aux_shape_{}", basename, tensor.index, i);
        shape.op = "Shape".into();
        shape.device = devices[i].clone();
        // shape.attr.insert("T".into(), attr(AttrValue_oneof_value::field_type(DataType::DT_INT32)));
        shape.attr.insert("T".into(), dtype.clone());
        shape.input.push(list[i].1.clone());
        target.pb.node.push(shape);
        format!("{}/ring_{}/aux_shape_{}", basename, tensor.index, i)
    }).collect();

    // 2. flattening
    let flats: Vec<_> = (0..n).map(|i| {
        let mut shape = NodeDef::new();
        shape.name = format!("{}/ring_{}/aux_flat_{}/shape", basename, tensor.index, i);
        shape.op = "Const".into();
        shape.device = devices[i].clone();
        shape.attr.insert("dtype".into(),
            attr(AttrValue_oneof_value::field_type(DataType::DT_INT32)));
        let mut value = TensorProto::new();
        let mut x = crate::proto::tensor_shape::TensorShapeProto::new();
        let mut dim = crate::proto::tensor_shape::TensorShapeProto_Dim::new();
        dim.size = 1;
        x.dim.push(dim);
        value.dtype = DataType::DT_INT32;
        value.tensor_shape = protobuf::SingularPtrField::some(x);
        value.int_val.push(-1);
        shape.attr.insert("value".into(),
            attr(AttrValue_oneof_value::tensor(value)));
        target.pb.node.push(shape);

        let mut flat = NodeDef::new();
        flat.name = format!("{}/ring_{}/aux_flat_{}", basename, tensor.index, i);
        flat.op = "Reshape".into();
        flat.device = devices[i].clone();
        flat.attr.insert("T".into(), dtype.clone());
        flat.input.push(list[i].1.clone());
        flat.input.push(format!("{}/ring_{}/aux_flat_{}/shape", basename, tensor.index, i));
        target.pb.node.push(flat);
        format!("{}/ring_{}/aux_flat_{}", basename, tensor.index, i)
    }).collect();

    // 3. chunking
    let mut chunks: Vec<Vec<String>> = (0..n).map(|i| {
        let mut dim = NodeDef::new();
        dim.name = format!("{}/ring_{}/aux_split_{}/split_dim", basename, tensor.index, i);
        dim.op = "Const".into();
        dim.device = devices[i].clone();
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
        split.device = devices[i].clone();
        split.input.push(format!("{}/ring_{}/aux_split_{}/split_dim", basename, tensor.index, i));
        split.input.push(flats[i].clone());
        split.attr.insert("T".into(), dtype.clone());
        split.attr.insert("num_split".into(),
            attr(AttrValue_oneof_value::i(n.try_into().unwrap())));
        target.pb.node.push(split);

        (0..n).map(|j| {
            format!("{}/ring_{}/aux_split_{}:{}", basename, tensor.index, i, j)
        }).collect()
    }).collect();

    // 4. n-1 rounds of reducing. the last modified chunks (i+n-2) have the full content
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

    // 5. n-1 rounds of gathering
    for round in 0..n-1 {
        for i in 0..n {
            let mut identity = NodeDef::new();
            identity.name = format!("{}/ring_{}/aux_identity_{}_{}", basename, tensor.index, i, round);
            identity.op = "Identity".into();
            identity.device = devices[i].clone();
            identity.attr.insert("T".into(), dtype.clone());
            identity.input.push(chunks[(i+1) % n][(i+round+n-1) % n].clone());
            chunks[i][(i+round+n-1) % n] = identity.name.clone();
            target.pb.node.push(identity);
        }
    }

    // 6. concating
    let concated: Vec<_> = chunks.into_iter().enumerate().map(|(i, chunk)| {
        let mut axis = NodeDef::new();
        axis.name = format!("{}/ring_{}/aux_concat_{}/axis", basename, tensor.index, i);
        axis.op = "Const".into();
        axis.device = devices[i].clone();
        axis.attr.insert("dtype".into(), attr(AttrValue_oneof_value::field_type(DataType::DT_INT32)));
        let mut value = TensorProto::new();
        let shape = crate::proto::tensor_shape::TensorShapeProto::new();
        value.dtype = DataType::DT_INT32;
        value.tensor_shape = protobuf::SingularPtrField::some(shape);
        value.int_val.push(0);
        axis.attr.insert("value".into(), attr(AttrValue_oneof_value::tensor(value)));
        target.pb.node.push(axis);

        let mut concat = NodeDef::new();
        concat.name = format!("{}/ring_{}/aux_concat_{}", basename, tensor.index, i);
        concat.op = "ConcatV2".into();
        concat.device = devices[i].clone();
        concat.input = chunk.into_iter().collect();
        concat.input.push(format!("{}/ring_{}/aux_concat_{}/axis", basename, tensor.index, i));
        concat.attr.insert("N".into(), attr(AttrValue_oneof_value::i(n.try_into().unwrap())));
        concat.attr.insert("T".into(), dtype.clone());
        concat.attr.insert("Tidx".into(), attr(AttrValue_oneof_value::field_type(DataType::DT_INT32)));
        target.pb.node.push(concat);

        format!("{}/ring_{}/aux_concat_{}", basename, tensor.index, i)
    }).collect();

    // 7. restore shapes
    concated.into_iter().zip(shapes).enumerate().map(|(i, (concat, shape))| {
        let mut reshape = NodeDef::new();
        reshape.name = format!("{}/ring_{}/aux_reshape_{}", basename, tensor.index, i);
        reshape.op = "Reshape".into();
        reshape.device = devices[i].clone();
        reshape.attr.insert("T".into(), dtype.clone());
        reshape.input.push(concat);
        reshape.input.push(shape);
        target.pb.node.push(reshape);
        format!("{}/ring_{}/aux_reshape_{}", basename, tensor.index, i)
    }).collect()
}

fn all_reduce_sum_ring_chunked(tensor: &mut Tensor, target: &mut Target) {
    assert!(tensor.node().replicated().unwrap());

    let list: Vec<_> = tensor.node().replicas.iter().map(|(id, name)| (*id, format!("{}:{}", name, tensor.index))).collect();
    let results = _all_reduce_sum_ring_chunked(tensor, &list, target);

    assert!(Iterator::eq(list.iter().map(|(x, _)| *x), 0..target.devices.len()));
    tensor.replicated = Some(Box::new(move |id| results[id].clone())); // assuming id are 0-n
}

fn attr(v: AttrValue_oneof_value) -> AttrValue {
    let mut a = AttrValue::new();
    a.value = Some(v);
    a
}

fn get_dtype(x: &NodeDef) -> AttrValue {
    x.attr.get("dtype").or_else(|| x.attr.get("T")).unwrap().clone() // TODO: not really correct. Need to parse ops.pbtxt and follow type or type_attr.
}
