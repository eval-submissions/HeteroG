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
        for node in graph.nodes.iter_mut() {
            match &node.raw_node.name[..] {
                "VariableV2" => {
                    node.replicas.push((0, node.raw_node.name.clone()));
                    replicate_cache(node.get_output(0), target);
                }
                "Placeholder" => {
                    node.replicas.push((0, node.raw_node.name.clone()));
                    replicate_split(node.get_output(0), target);
                }
                _ => {
                    replicate_per_device(node, target);
                }
            }
        }
    }
}

fn replicate_per_device(node: &mut Node, target: &mut Target) {
    for i in 0..target.devices.len() {
        node.replicas.push((i, format!("{}/replica_{}", node.raw_node.name, i)))
    }
}

fn replicate_split(tensor: &mut Tensor, target: &mut Target) {
    assert!(tensor.node().replicas.len() == 1);

    let name = tensor.node().raw_node.name.clone();
    let index = tensor.index; // clone here because we will move it later

    let mut dim = NodeDef::new();
    dim.name = format!("{}/aux_split_{}/split_dim", name, index);
    dim.op = "Const".into();
    dim.device = target.devices[tensor.node().replicas[0].0].clone();
    dim.input.push(tensor.original_name());
    dim.attr.insert("dtype".into(),
        attr(AttrValue_oneof_value::field_type(DataType::DT_FLOAT))); // TODO: get from raw_node
    let value = TensorProto::new();
    let shape = crate::proto::tensor_shape::TensorShapeProto::new();
    let scalar = crate::proto::tensor_shape::TensorShapeProto_Dim::new();
    shape.dim.push(scalar);

    value.dtype = DataType::DT_FLOAT; // TODO: get from raw_node
    value.tensor_shape = protobuf::SingularPtrField::some(shape);
    value.int_val.push(0);
    dim.attr.insert("value".into(),
        attr(AttrValue_oneof_value::tensor(TensorProto::new())));

    // create_raw_node(target, 'Const', '',
    //     "{}/{}".format(self.node.raw_node.name, "aux_split_{}/split_dim".format(self.index)),
    //     dtype = { 'type': DT_INT32 },
    //     value = { 'tensor': { 'dtype': DT_INT32, 'tensor_shape': {}, 'int_val': [0] } }
    // )
    // create_raw_node(target, 'Split', '',
    //     "{}/{}".format(self.node.raw_node.name, "aux_split_{}".format(self.index)),
    //     "{}/{}".format(self.node.raw_node.name, "aux_split_{}/split_dim".format(self.index)),
    //     "{}:{}".format(self.node.replicas[0].name, self.index),
    //     T = { 'type': DT_FLOAT },
    //     num_split = { 'i': len(self.node.graph.devices) }
    // )
}

/// direct identity node: no topology and routing considered
fn replicate_cache(tensor: &mut Tensor, target: &mut Target) {
    assert!(tensor.node().replicas.len() == 1);

    let name = tensor.node().raw_node.name.clone();
    let index = tensor.index; // clone here because we will move it later

    for (id, device) in target.devices.iter().enumerate() {
        let mut identity = NodeDef::new();

        identity.name = format!("{}/aux_identity_{}_{}", name, index, id);
        identity.op = "Identity".into();
        identity.device = device.clone();

        let dtype = attr(AttrValue_oneof_value::field_type(DataType::DT_FLOAT)); // TODO: get from raw_node
        identity.attr.insert("T".into(), dtype);
        identity.input.push(tensor.original_name());

        target.pb.node.push(identity)
    }

    tensor.aggregated = Some(format!("{}:{}", name, index));
    tensor.replicated = Some(Box::new(move |id| format!("{}/aux_identity_{}_{}", name, index, id)));
}

fn attr(v: AttrValue_oneof_value) -> AttrValue {
    let mut a = AttrValue::new();
    a.value = Some(v);
    a
}
