use oh_my_rust::*;
use protobuf::Message;
use crate::proto::{graph::GraphDef, node_def::NodeDef, attr_value::AttrValue, types::DataType};
use std::collections::BTreeMap;
use std::fmt::Write;
use std::convert::TryInto;
use crate::strategy::Strategy;

pub struct Graph<NEX: Default, TEX: Default> {
    pub nodes: Vec<Node<NEX, TEX>>, // This vector is partial ordered: inputs are guarenteed to appear ealier than descendents
    pub name_dict: std::collections::BTreeMap<String, usize>
}

impl<NEX: Default, TEX: Default> Graph<NEX, TEX> {
    pub fn new(nodes: &[NodeDef]) -> Box<Self> {
        task!("build graph of {} nodes", nodes.len());

        let mut g = Box::new(Graph { nodes: Vec::with_capacity(nodes.len()), name_dict: BTreeMap::new() });

        // no always optimal, but good enough since the input is actually mostly ordered
        let mut queue: std::collections::VecDeque::<_> = nodes.iter().collect();
        'outer: while let Some(node_def) = queue.pop_front() {
            for input in node_def.input.iter() {
                let input = if input.starts_with('^') {
                    &input[1..]
                } else {
                    parse_input(input).0
                };
                if !g.name_dict.contains_key(input) {
                    debug!("pushing back {}", node_def.name);
                    queue.push_back(node_def);
                    continue 'outer;
                }
            }

            let node = Node::new(&g, node_def.clone());
            g.name_dict.insert(node.raw_node.name.clone(), g.nodes.len());
            g.nodes.push(node);
        }

        g
    }

    /// setup the replicas and links. Note that auxiliary nodes are already there by strategies.
    /// The only required fields is `replicas`. All other decisions are optional.
    pub fn compile(&mut self, target: &mut Target) {
        task!("compile graph of {} nodes", self.nodes.len());
        for node in self.nodes.iter_mut() {
            node.compile(target)
        }
    }
}

/// Cache means "full". Split means splitted by the batch size dimension.
#[derive(Debug, Eq, PartialEq, Clone, Copy)]
pub enum ReplicationType { Cache, Split }

pub struct Node<NEX: Default, TEX: Default> {
    pub graph: *const Graph<NEX, TEX>,
    pub raw_node: NodeDef,
    pub controls: Vec<usize>, // TODO: more consideration for control dependencies that added aux nodes
    pub inputs: Vec<(usize, usize)>, // nodeid, index
    pub outputs: Vec<Tensor<NEX, TEX>>,

    pub replicas: Vec<(usize, String)>, // deviceid, name. This must be filled by strategies.
    pub input_replication_types: Box<[ReplicationType]>, // the replication type of each input. This must be filled by strategies.

    pub extra: NEX
}

impl<NEX: Default, TEX: Default> Node<NEX, TEX> {
    pub fn new(graph: &Graph<NEX, TEX>, raw_node: NodeDef) -> Self {
        let mut inputs = vec![];
        let mut controls = vec![];

        for input in raw_node.input.iter() {
            if input.starts_with('^') {
                controls.push(graph.name_dict[&input[1..]])
            } else {
                let (name, index) = parse_input(input);
                let id = graph.name_dict[name];
                inputs.push((id, index))
            }
        }

        let default_replication_type = inputs.iter().map(|_| ReplicationType::Cache).collect();

        Self {
            graph, raw_node, controls, inputs,
            outputs: vec![], replicas: vec![],
            input_replication_types: default_replication_type,
            extra: Default::default()
        }
    }

    #[allow(clippy::mut_from_ref)]
    pub fn graph<'a>(&self) -> &'a mut Graph<NEX, TEX> {
        unsafe { &mut *(self.graph as *mut Graph<NEX, TEX>) }
    }

    #[allow(clippy::mut_from_ref, clippy::cast_ref_to_mut)]
    pub fn get_output(&self, index: usize) -> &mut Tensor<NEX, TEX> {
        // TODO: use Interior mutable for outputs?
        let mutable = unsafe { &mut *(self as *const Node<NEX, TEX> as *mut Node<NEX, TEX>) };

        while mutable.outputs.len() <= index {
            mutable.outputs.push(Tensor::new(mutable, mutable.outputs.len()))
        }

        &mut mutable.outputs[index]
    }

    pub fn replicated(&self) -> Option<bool> {
        match self.replicas.len() {
            0 => None,
            1 => Some(false),
            _ => Some(true)
        }
    }

    // if any of the inputs are splitted, so if it is replicated, the outputs can be treat as splitted
    pub fn splitted(&self) -> bool {
        self.input_replication_types.iter().any(|x| x == &ReplicationType::Split)
    }

    /// add an edited node into the target. Requires all inputs to be compiled first
    fn compile(&mut self, target: &mut Target) {
        debug!("compile: {} {:?} {:?}", self.raw_node.name, self.input_replication_types, self.replicas.iter().map(|x| x.0).collect::<Vec<_>>());

        for (device_id, name) in self.replicas.iter() {
            // 1. setup basic node info
            let mut node = self.raw_node.clone();
            node.name = name.clone();
            node.device = target.devices[*device_id].clone();
            node.attr.insert("_tge_origin".into(), AttrValue::new().apply_owned(|x| x.set_s(self.raw_node.name.clone().into_bytes())));

            // 2. link inputs
            node.input = self.inputs.iter().zip(self.input_replication_types.iter()).map(|((node_id, index), reptype)| {
                let input_tensor = &mut self.graph().nodes[*node_id].get_output(*index);
                match reptype {
                    ReplicationType::Cache => input_tensor.get_cache(*device_id, target),
                    ReplicationType::Split => input_tensor.get_split(*device_id, target)
                }
            }).collect();

            // 3. add control dependencies
            for node_id in self.controls.iter() {
                let dep_node = &self.graph().nodes[*node_id];
                let dependency = if dep_node.replicated().unwrap() {
                    &dep_node.replicas[*device_id].1
                } else {
                    &dep_node.replicas[0].1
                };
                node.input.push(format!("^{}", dependency))
            }

            target.pb.node.push(node)
        }
    }

    /**************************************
    * following are graph editing methods *
    **************************************/

    pub fn make_node(&self, op: String) -> NodeDef {
        let mut node = NodeDef::new();
        node.op = op;
        node.name = self.raw_node.name.clone();
        node.attr.insert("_tge_belong_to".into(), AttrValue::new().apply_owned(|x| x.set_s(self.raw_node.name.clone().into_bytes())));
        node
    }

    pub fn put_on_device(&mut self, device_id: usize) {
        assert!(self.replicated().is_none(), "already replicated!");
        self.replicas.push((device_id, self.raw_node.name.clone()));
    }

    pub fn put_on_all_devices(&mut self, target: &mut Target) {
        assert!(self.replicated().is_none(), "already replicated!");
        let name = &self.raw_node.name;
        self.replicas.extend((0..target.devices.len()).map(|i| (i, format!("{}/replica_{}", name, i))));
    }
}

pub struct Tensor<NEX: Default, TEX: Default> {
    pub node: *const Node<NEX, TEX>,
    pub index: usize,

    pub cache: Vec<String>, // may be provided by strategy
    pub split: Vec<String>, // may be provided by strategy

    pub extra: TEX,
}

impl<NEX: Default, TEX: Default> Tensor<NEX, TEX> {
    pub fn new(node: &Node<NEX, TEX>, index: usize) -> Self {
        Tensor { node, index, cache: vec![], split: vec![], extra: TEX::default() }
    }

    pub fn original_name(&self) -> String {
        if self.index == 0 {
            self.node().raw_node.name.clone()
        } else {
            format!("{}:{}", self.node().raw_node.name, self.index)
        }
    }

    pub fn node<'a>(&self) -> &'a Node<NEX, TEX> {
        unsafe { &*self.node }
    }

    pub fn get_shape(&self) -> Vec<usize> {
        // sucks: the output shape of BroadcastGradientArgs is always unknown even if inputs are fixed
        // and ops like `Sum` (requires the dimension to sum along with) and `Fill` operates differenct with different input
        self.node().raw_node.attr["_output_shapes"].get_list().shape[self.index].dim.iter().map(|x| x.size.try_into().ok()).collect::<Option<_>>().unwrap_or_else(Vec::new)
    }

    pub fn get_cache(&mut self, device_id: usize, target: &mut Target) -> String {
        if self.cache.is_empty() {
            if self.node().replicated().unwrap() {
                if self.node().splitted() {
                    self.aggregate_cat(device_id, target);
                } else {
                    self.cache.extend_from_slice(&self.node().replicas.iter().map(|(_, name)| format!("{}:{}", name, self.index)).collect::<Vec<_>>());
                }
            } else {
                let name = format!("{}:{}", self.node().replicas[0].1, self.index);
                self.cache.extend((0..target.devices.len()).map(|_| name.clone()));
            }
        }

        self.cache[device_id].clone()
    }

    pub fn get_split(&mut self, device_id: usize, target: &mut Target) -> String {
        if self.split.is_empty() {
            if self.node().replicated().unwrap() {
                if self.node().splitted() {
                    self.split.extend_from_slice(&self.node().replicas.iter().map(|(_, name)| format!("{}:{}", name, self.index)).collect::<Vec<_>>());
                } else {
                    unimplemented!();
                }
            } else {
                self.replicate_split(target);
            }
        }

        self.split[device_id].clone()
    }

    /**************************************
    * following are graph editing methods *
    **************************************/

    pub fn aggregate_sum(&mut self, server: usize, target: &mut Target) {
        assert!(self.node().replicated().unwrap() && self.node().splitted() && self.cache.is_empty());

        let mut addn = self.node().make_node("AddN".to_string());
        addn.name += &format!("/aux_sum_{}", self.index);
        addn.device = target.devices[server].clone();
        addn.attr.insert("N".into(), AttrValue::new().apply_owned(|x| x.set_i(self.node().replicas.len().try_into().unwrap())));
        addn.attr.insert("T".into(), get_dtype(&self.node().raw_node));
        addn.input = self.node().replicas.iter().map(|(_, x)| format!("{}:{}", x, self.index)).collect();

        self.cache.extend((0..target.devices.len()).map(|_| addn.name.clone()));
        target.pb.node.push(addn);
    }

    // TODO: share the same axis nodes for all concating (and do the same thing for dim nodes in splitting)

    pub fn aggregate_cat(&mut self, server: usize, target: &mut Target) {
        assert!(self.node().replicated().unwrap() && self.node().splitted() && self.cache.is_empty());

        let mut axis = self.node().make_node("Const".to_string());
        axis.name += &format!("/aux_concat_{}/axis", self.index);
        axis.device = target.devices[server].clone();
        axis.attr.insert("dtype".into(), AttrValue::new().apply_owned(|x| x.set_field_type(DataType::DT_INT32)));
        let value = crate::proto::tensor::TensorProto::new().apply_owned(|x| {
            x.set_dtype(DataType::DT_INT32);
            x.set_tensor_shape(crate::proto::tensor_shape::TensorShapeProto::new());
            x.int_val.push(0);
        });
        axis.attr.insert("value".into(), AttrValue::new().apply_owned(|x| x.set_tensor(value)));

        let mut concat = self.node().make_node("ConcatV2".to_string());
        concat.name += &format!("/aux_concat_{}", self.index);
        concat.device = target.devices[server].clone();
        concat.input = self.node().replicas.iter().map(|(_, x)| format!("{}:{}", x, self.index)).collect();
        concat.input.push(format!("{}/aux_concat_{}/axis", self.node().raw_node.name, self.index));
        concat.attr.insert("N".into(), AttrValue::new().apply_owned(|x| x.set_i(self.node().replicas.len().try_into().unwrap())));
        concat.attr.insert("T".into(), get_dtype(&self.node().raw_node));
        concat.attr.insert("Tidx".into(), AttrValue::new().apply_owned(|x| x.set_field_type(DataType::DT_INT32)));

        self.cache.extend((0..target.devices.len()).map(|_| concat.name.clone()));
        target.pb.node.push(axis);
        target.pb.node.push(concat);
    }

    pub fn replicate_split(&mut self, target: &mut Target) {
        assert!(!self.node().replicated().unwrap() && self.split.is_empty());

        let mut dim = self.node().make_node("Const".to_string());
        dim.name += &format!("/aux_split_{}/split_dim", self.index);
        dim.device = target.devices[self.node().replicas[0].0].clone();
        dim.attr.insert("dtype".into(), AttrValue::new().apply_owned(|x| x.set_field_type(DataType::DT_INT32)));
        let value = crate::proto::tensor::TensorProto::new().apply_owned(|x| {
            x.set_dtype(DataType::DT_INT32);
            x.set_tensor_shape(crate::proto::tensor_shape::TensorShapeProto::new());
            x.int_val.push(0);
        });
        dim.attr.insert("value".into(), AttrValue::new().apply_owned(|x| x.set_tensor(value)));

        let mut split = self.node().make_node("Split".to_string());
        split.name += &format!("/aux_split_{}", self.index);
        split.device = target.devices[self.node().replicas[0].0].clone();
        split.input.push(dim.name.clone());
        split.input.push(format!("{}:{}", self.node().replicas[0].1, self.index));
        split.attr.insert("T".into(), get_dtype(&self.node().raw_node));
        split.attr.insert("num_split".into(), AttrValue::new().apply_owned(|x| x.set_i(target.devices.len().try_into().unwrap())));

        self.split.extend((0..target.devices.len()).map(|i| format!("{}:{}", split.name, i)));
        target.pb.node.push(dim);
        target.pb.node.push(split);
    }

    pub fn all_reduce_nccl(&mut self, target: &mut Target) {
        // to all_sum n tensors (can be on the same devie), one should have n NcclAllReduce nodes with the same shared_name attr
        // each node have only *one* input, and should be on the same device of the input. The output of these nodes will be the same

        assert!(self.node().replicated().unwrap() && self.node().splitted() && self.cache.is_empty());

        for (id, replica) in self.node().replicas.iter() {
            let mut nccl = self.node().make_node("NcclAllReduce".to_string());
            nccl.name += &format!("/aux_nccl_{}_{}", self.index, id);
            nccl.device = target.devices[*id].clone();
            nccl.attr.insert("reduction".into(), AttrValue::new().apply_owned(|x| x.set_s(b"sum".to_vec())));
            nccl.attr.insert("T".into(), get_dtype(&self.node().raw_node));
            nccl.attr.insert("num_devices".into(), AttrValue::new().apply_owned(|x| x.set_i(self.node().replicas.len().try_into().unwrap())));
            nccl.attr.insert("shared_name".into(), AttrValue::new().apply_owned(|x| x.set_s(self.original_name().into_bytes())));
            nccl.input.push(format!("{}:{}", replica, self.index));

            target.pb.node.push(nccl)
        }

        self.cache.extend_from_slice(&(0..target.devices.len()).map(|i| format!("{}/aux_nccl_{}_{}", self.node().raw_node.name, self.index, i)).collect::<Vec<_>>());
    }

    pub fn all_reduce_ring(&mut self, target: &mut Target) {
        assert!(self.node().replicated().unwrap() && self.node().splitted() && self.cache.is_empty());

        let list: Vec<_> = self.node().replicas.iter().map(|(id, name)| (*id, format!("{}:{}", name, self.index))).collect();
        let results = _all_reduce_sum_ring_chunked(self, &list, target);

        assert!(Iterator::eq(list.iter().map(|(x, _)| *x), 0..target.devices.len()));
        self.cache.extend((0..target.devices.len()).map(|i| results[i].clone()));
    }
}

pub struct Target {
    pub pb: GraphDef,
    pub devices: Box<[String]>,
    pub links: Box<[u64]>, // the bandwidth of each link
    pub paths: Box<[Box<[usize]>]> // the i*n+j element is the links that i->j uses (currently only one path between each pair)
}

impl Target {
    pub fn new(pb: GraphDef, devices: Box<[String]>, links: Box<[u64]>, paths: Box<[Box<[usize]>]>) -> Self {
        Target { pb, devices, links, paths }
    }
}

// TODO: This function is not done. Need to parse ops.pbtxt and follow type or type_attr.
fn get_dtype(x: &NodeDef) -> AttrValue {
    match &x.op[..] {
        "Greater" | "GreaterEqual" => AttrValue::new().apply_owned(|x| x.set_field_type(DataType::DT_BOOL)),
        "Shape" | "ShapeN" => x.attr.get("out_type").cloned().unwrap_or_else(|| AttrValue::new().apply_owned(|x| x.set_field_type(DataType::DT_INT32))),
        "Cast" => x.attr.get("DstT").cloned().unwrap(),
        _ => x.attr.get("dtype").or_else(|| x.attr.get("T")).unwrap_or_else(|| panic!("cannot determine dtype for {}", x.op)).clone()
    }
}

fn parse_input(x: &str) -> (&str, usize) {
    match x.find(':') {
        Some(i) => (&x[..i], x[i+1..].parse().unwrap()),
        None => (x, 0)
    }
}

/// performing chunked ring all reduce for a list of (device_id, tensor_name), returning the name of summed results on each device
fn _all_reduce_sum_ring_chunked<NEX: Default, TEX: Default>(tensor: &Tensor<NEX, TEX>, list: &[(usize, String)], target: &mut Target) -> Vec<String> {
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
        shape.attr.insert("dtype".into(), AttrValue::new().apply_owned(|x| x.set_field_type(DataType::DT_INT32)));
        let mut value = crate::proto::tensor::TensorProto::new();
        let mut x = crate::proto::tensor_shape::TensorShapeProto::new();
        let mut dim = crate::proto::tensor_shape::TensorShapeProto_Dim::new();
        dim.size = 1;
        x.dim.push(dim);
        value.dtype = DataType::DT_INT32;
        value.tensor_shape = protobuf::SingularPtrField::some(x);
        value.int_val.push(-1);
        shape.attr.insert("value".into(), AttrValue::new().apply_owned(|x| x.set_tensor(value)));
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
        dim.attr.insert("dtype".into(), AttrValue::new().apply_owned(|x| x.set_field_type(DataType::DT_INT32)));
        let mut value = crate::proto::tensor::TensorProto::new();
        let shape = crate::proto::tensor_shape::TensorShapeProto::new();
        value.dtype = DataType::DT_INT32;
        value.tensor_shape = protobuf::SingularPtrField::some(shape);
        value.int_val.push(0);
        dim.attr.insert("value".into(), AttrValue::new().apply_owned(|x| x.set_tensor(value)));
        target.pb.node.push(dim);

        let mut split = NodeDef::new();
        split.name = format!("{}/ring_{}/aux_split_{}", basename, tensor.index, i);
        split.op = "Split".into();
        split.device = devices[i].clone();
        split.input.push(format!("{}/ring_{}/aux_split_{}/split_dim", basename, tensor.index, i));
        split.input.push(flats[i].clone());
        split.attr.insert("T".into(), dtype.clone());
        split.attr.insert("num_split".into(), AttrValue::new().apply_owned(|x| x.set_i(n.try_into().unwrap())));
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
        axis.attr.insert("dtype".into(), AttrValue::new().apply_owned(|x| x.set_field_type(DataType::DT_INT32)));
        let mut value = crate::proto::tensor::TensorProto::new();
        let shape = crate::proto::tensor_shape::TensorShapeProto::new();
        value.dtype = DataType::DT_INT32;
        value.tensor_shape = protobuf::SingularPtrField::some(shape);
        value.int_val.push(0);
        axis.attr.insert("value".into(), AttrValue::new().apply_owned(|x| x.set_tensor(value)));
        target.pb.node.push(axis);

        let mut concat = NodeDef::new();
        concat.name = format!("{}/ring_{}/aux_concat_{}", basename, tensor.index, i);
        concat.op = "ConcatV2".into();
        concat.device = devices[i].clone();
        concat.input = chunk.into_iter().collect();
        concat.input.push(format!("{}/ring_{}/aux_concat_{}/axis", basename, tensor.index, i));
        concat.attr.insert("N".into(), AttrValue::new().apply_owned(|x| x.set_i(n.try_into().unwrap())));
        concat.attr.insert("T".into(), dtype.clone());
        concat.attr.insert("Tidx".into(), AttrValue::new().apply_owned(|x| x.set_field_type(DataType::DT_INT32)));
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
