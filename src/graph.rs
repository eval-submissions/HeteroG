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
        task!("building graph of {} nodes...", nodes.len());

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
    pub fn compile(&mut self, target: &mut Target) {
        task!("compiling graph of {} nodes...", self.nodes.len());
        for node in self.nodes.iter_mut() {
            node.compile(target)
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd, Copy)]
pub enum FormKind { Full, Part }

#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd)]
pub struct Form {
    pub kind: FormKind,
    pub devices: Vec<usize> // use Vec before got a fancy bit set. The Vec must be sorted and not empty.
}

impl Form {
    pub fn is_full(&self) -> bool {
        self.kind == FormKind::Full
    }

    pub fn is_part(&self) -> bool {
        self.kind == FormKind::Part
    }
}

pub struct Node<NEX: Default, TEX: Default> {
    pub graph: *const Graph<NEX, TEX>,
    pub raw_node: NodeDef,
    pub controls: Vec<usize>, // TODO: more consideration for control dependencies that added aux nodes
    pub inputs: Vec<(usize, usize, FormKind)>, // nodeid, index, formkind (defaults to full)
    pub outputs: Vec<Tensor<NEX, TEX>>,
    pub form: Form, // the form of the node, which is also a tensor form for all its outputs

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
                inputs.push((id, index, FormKind::Full))
            }
        }

        Self {
            graph, raw_node, controls, inputs, outputs: vec![],
            form: Form { kind: FormKind::Full, devices: vec![] },
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
        match self.form.devices.len() {
            0 => None,
            1 => Some(false),
            _ => Some(true)
        }
    }

    /// add an edited node into the target. Requires all inputs to be compiled first
    fn compile(&mut self, target: &mut Target) {
        debug!("compile: {} {:?} {:?}", self.raw_node.name, self.form, self.inputs);

        for (replica_index, device_id) in self.form.devices.iter().enumerate() {
            // 1. setup basic node info
            let mut node = self.raw_node.clone();
            node.name = self.replica_name_on_device(*device_id);
            node.device = target.devices[*device_id].clone();
            set_origin(&mut node, &self.raw_node.name);

            // 2. link inputs and set size
            node.input = self.inputs.iter().copied().enumerate().map(|(i, (node_id, index, kind))| {
                let input_tensor = &mut self.graph().nodes[node_id].get_output(index);
                set_input_size(&mut node, i, match self.form.kind {
                    FormKind::Full => input_tensor.get_size(),
                    FormKind::Part => input_tensor.get_size() / self.form.devices.len() as u64,
                });
                let input_names = input_tensor.as_form(&Form { kind, devices: self.form.devices.clone() });
                input_names[replica_index].clone()
            }).collect();

            // 3. add control dependencies
            for node_id in self.controls.iter() {
                let dep_node = &self.graph().nodes[*node_id];
                for device_id in dep_node.form.devices.iter() {
                    node.input.push(dep_node.replica_name_on_device(*device_id))
                }
            }

            target.pb.node.push(node)
        }
    }

    fn replica_name_on_device(&self, device_id: usize) -> String { // TODO: should this method exist?
        format!("{}/replica_{}", self.raw_node.name, device_id)
    }

    /**************************************
    * following are graph editing methods *
    **************************************/

    pub fn make_node(&self, op: String) -> NodeDef {
        let mut node = NodeDef::new();
        node.op = op;
        node.name = self.raw_node.name.clone();
        set_belong_to(&mut node, &self.raw_node.name);
        node
    }

    pub fn put_on_devices(&mut self, devices: &[usize]) {
        assert!(self.replicated().is_none(), "already set replicas!");
        self.form.devices.extend_from_slice(devices);
    }
}

pub struct Tensor<NEX: Default, TEX: Default> {
    pub node: *const Node<NEX, TEX>,
    pub index: usize,
    pub forms: BTreeMap<Form, Box<[String]>>,

    pub extra: TEX,
}

impl<NEX: Default, TEX: Default> Tensor<NEX, TEX> {
    pub fn new(node: &Node<NEX, TEX>, index: usize) -> Self {
        Tensor { node, index, forms: BTreeMap::new(), extra: TEX::default() }
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
        // and ops like `Sum` (requires the dimension to sum along with) and `Fill` operates differently with different inputs
        self.node().raw_node.attr["_output_shapes"].get_list().shape[self.index].dim.iter().map(|x| x.size.try_into().ok()).collect::<Option<_>>().unwrap_or_else(Vec::new)
    }

    pub fn get_size(&self) -> u64 {
        #[allow(clippy::unnecessary_fold)]
        (self.get_shape().iter().fold(1, |x, y| x * y) * 4).try_into().unwrap()
    }

    // get the names as the specified form
    pub fn as_form(&mut self, form: &Form) -> &[String] {
        if !self.forms.contains_key(form) {
            let names = if form == &self.node().form {
                form.devices.iter().map(|device_id| format!("{}:{}", self.node().replica_name_on_device(*device_id), self.index)).collect()
            } else {
                let node_kind = self.node().form.kind;
                match (form.kind, node_kind) {
                    (FormKind::Full, FormKind::Full) => { // easy mode, use the copy in the corresponding device, or from the first device otherwise
                        let raw = self.as_form(&self.node().form).to_vec(); // TODO: no clone?
                        form.devices.iter().map(|device_id| {
                            self.node().form.devices.iter().position(|x| *x == *device_id).map(|ind| raw[ind].clone()).unwrap_or(raw[0].clone())
                        }).collect()
                    }
                    _ => unimplemented!()
                }
            };

            self.forms.insert(form.clone(), names);
        }

        &self.forms[form]
    }

    /**************************************
    * following are graph editing methods *
    **************************************/

    pub fn aggregate_sum(&mut self, from: &Form, to: &Form, target: &mut Target) -> Box<[String]> {
        assert!(from.is_part() && to.is_full() && to.devices.len() == 1);

        let mut addn = self.node().make_node("AddN".to_string());
        addn.name += &format!("/aux_sum_{}", self.index);
        addn.device = target.devices[to.devices[0]].clone();
        addn.attr.insert("N".into(), AttrValue::new().apply(|x| x.set_i(from.devices.len().try_into().unwrap())));
        addn.attr.insert("T".into(), get_dtype(&self.node().raw_node));
        addn.input = self.as_form(from).iter().cloned().collect();
        for i in 0..from.devices.len() {
            set_input_size(&mut addn, i, self.get_size() / from.devices.len() as u64)
        }

        let result = vec![addn.name.clone()].into_boxed_slice();
        target.pb.node.push(addn);
        result
    }

    // TODO: share the same axis nodes for all concating (and do the same thing for dim nodes in splitting)
    pub fn aggregate_cat(&mut self, from: &Form, to: &Form, target: &mut Target) -> Box<[String]> {
        assert!(from.is_part() && to.is_full() && to.devices.len() == 1);

        let mut axis = self.node().make_node("Const".to_string());
        axis.name += &format!("/aux_concat_{}/axis", self.index);
        axis.device = target.devices[to.devices[0]].clone();
        axis.attr.insert("dtype".into(), AttrValue::new().apply(|x| x.set_field_type(DataType::DT_INT32)));
        let value = crate::proto::tensor::TensorProto::new().apply(|x| {
            x.set_dtype(DataType::DT_INT32);
            x.set_tensor_shape(crate::proto::tensor_shape::TensorShapeProto::new());
            x.int_val.push(0);
        });
        axis.attr.insert("value".into(), AttrValue::new().apply(|x| x.set_tensor(value)));

        let mut concat = self.node().make_node("ConcatV2".to_string());
        concat.name += &format!("/aux_concat_{}", self.index);
        concat.device = target.devices[to.devices[0]].clone();
        concat.input = self.as_form(from).iter().cloned().collect();
        concat.input.push(format!("{}/aux_concat_{}/axis", self.node().raw_node.name, self.index));
        concat.attr.insert("N".into(), AttrValue::new().apply(|x| x.set_i(from.devices.len().try_into().unwrap())));
        concat.attr.insert("T".into(), get_dtype(&self.node().raw_node));
        concat.attr.insert("Tidx".into(), AttrValue::new().apply(|x| x.set_field_type(DataType::DT_INT32)));
        for i in 0..from.devices.len() {
            set_input_size(&mut concat, i, self.get_size() / from.devices.len() as u64)
        }

        let result = vec![concat.name.clone()].into_boxed_slice();
        target.pb.node.push(axis);
        target.pb.node.push(concat);
        result
    }

    // currenly we only split from the first replica. Future we can split on every device and use the local copy to reduce transfering
    pub fn replicate_split(&mut self, from: &Form, to: &Form, target: &mut Target) -> Box<[String]> {
        assert!(from.is_full() && to.is_part());

        let mut dim = self.node().make_node("Const".to_string());
        dim.name += &format!("/aux_split_{}/split_dim", self.index);
        dim.device = target.devices[from.devices[0]].clone();
        dim.attr.insert("dtype".into(), AttrValue::new().apply(|x| x.set_field_type(DataType::DT_INT32)));
        let value = crate::proto::tensor::TensorProto::new().apply(|x| {
            x.set_dtype(DataType::DT_INT32);
            x.set_tensor_shape(crate::proto::tensor_shape::TensorShapeProto::new());
            x.int_val.push(0);
        });
        dim.attr.insert("value".into(), AttrValue::new().apply(|x| x.set_tensor(value)));

        let mut split = self.node().make_node("Split".to_string());
        split.name += &format!("/aux_split_{}", self.index);
        split.device = target.devices[from.devices[0]].clone();
        split.input.push(dim.name.clone());
        split.input.push(format!("{}:{}", self.as_form(from)[0].clone(), self.index));
        split.attr.insert("T".into(), get_dtype(&self.node().raw_node));
        split.attr.insert("num_split".into(), AttrValue::new().apply(|x| x.set_i(to.devices.len().try_into().unwrap())));
        set_input_size(&mut split, 1, self.get_size());

        let result = (0..to.devices.len()).map(|i| format!("{}:{}", split.name, i)).collect();
        target.pb.node.push(dim);
        target.pb.node.push(split);
        result
    }

    pub fn all_reduce_nccl(&mut self, from: &Form, to: &Form, target: &mut Target) -> Box<[String]> {
        // to all_sum n tensors (can be on the same devie), one should have n NcclAllReduce nodes with the same shared_name attr
        // each node have only *one* input, and should be on the same device of the input. The output of these nodes will be the same

        assert!(from.is_part() && to.is_full() && from.devices == to.devices);

        let index = self.index;

        for (i, device_id) in from.devices.iter().copied().enumerate() {
            let mut nccl = self.node().make_node("NcclAllReduce".to_string());
            nccl.name += &format!("/aux_nccl_{}_{}", index, i);
            nccl.device = target.devices[device_id].clone();
            nccl.attr.insert("reduction".into(), AttrValue::new().apply(|x| x.set_s(b"sum".to_vec())));
            nccl.attr.insert("T".into(), get_dtype(&self.node().raw_node));
            nccl.attr.insert("num_devices".into(), AttrValue::new().apply(|x| x.set_i(from.devices.len().try_into().unwrap())));
            nccl.attr.insert("shared_name".into(), AttrValue::new().apply(|x| x.set_s(self.original_name().into_bytes())));
            nccl.input.push(format!("{}:{}", self.as_form(from)[i], index));

            target.pb.node.push(nccl)
        }

        (0..from.devices.len()).map(|i| format!("{}/aux_nccl_{}_{}", self.node().raw_node.name, self.index, i)).collect()
    }

    // pub fn all_reduce_ring(&mut self, target: &mut Target) {
    //     assert!(self.node().replicated().unwrap() && self.node().splitted() && self.cache.is_empty());

    //     let list: Vec<_> = self.node().replicas.iter().map(|(id, name)| (*id, format!("{}:{}", name, self.index))).collect();
    //     let results = _all_reduce_sum_ring_chunked(self, &list, target);

    //     assert!(Iterator::eq(list.iter().map(|(x, _)| *x), 0..target.devices.len()));
    //     self.cache.extend((0..target.devices.len()).map(|i| results[i].clone()));
    // }
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

fn set_origin(node: &mut NodeDef, origin: &str) {
    node.attr.insert("_tge_origin".into(), AttrValue::new().apply(|x| x.set_s(origin.as_bytes().to_vec())));
}

fn set_belong_to(node: &mut NodeDef, belong_to: &str) {
    node.attr.insert("_tge_belong_to".into(), AttrValue::new().apply(|x| x.set_s(belong_to.as_bytes().to_vec())));
}

fn set_input_size(node: &mut NodeDef, index: usize, size: u64) {
    let sizes = &mut node.attr.entry("_tge_input_sizes".to_string()).or_insert_with(AttrValue::new).mut_list().i;
    if sizes.len() <= index {
        sizes.resize(index+1, 0)
    }
    sizes[index] = size as _;
}

// TODO: This function is not done. Need to parse ops.pbtxt and follow type or type_attr.
fn get_dtype(x: &NodeDef) -> AttrValue {
    match &x.op[..] {
        "Greater" | "GreaterEqual" => AttrValue::new().apply(|x| x.set_field_type(DataType::DT_BOOL)),
        "Shape" | "ShapeN" => x.attr.get("out_type").cloned().unwrap_or_else(|| AttrValue::new().apply(|x| x.set_field_type(DataType::DT_INT32))),
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

// /// performing chunked ring all reduce for a list of (device_id, tensor_name), returning the name of summed results on each device
// fn _all_reduce_sum_ring_chunked<NEX: Default, TEX: Default>(tensor: &Tensor<NEX, TEX>, list: &[(usize, String)], target: &mut Target) -> Vec<String> {
//     let n = list.len();
//     let basename = tensor.node().raw_node.name.clone();
//     let devices: Vec<_> = list.iter().map(|(id, _)| target.devices[*id].clone()).collect();
//     let dtype = get_dtype(&tensor.node().raw_node);
//     let psize = tensor.get_size() / tensor.node().replicas.len() as u64;

//     // 1. recording the shape
//     let shapes: Vec<_> = (0..n).map(|i| {
//         let mut shape = tensor.node().make_node("Shape".to_string());
//         shape.name += &format!("/ring_{}/aux_shape_{}", tensor.index, i);
//         shape.device = devices[i].clone();
//         shape.attr.insert("T".into(), dtype.clone());
//         shape.input.push(list[i].1.clone());
//         set_input_size(&mut shape, 0, psize);
//         target.pb.node.push(shape);
//         format!("{}/ring_{}/aux_shape_{}", basename, tensor.index, i)
//     }).collect();

//     // 2. flattening
//     let flats: Vec<_> = (0..n).map(|i| {
//         let mut shape = tensor.node().make_node("Const".to_string());
//         shape.name += &format!("/ring_{}/aux_flat_{}/shape", tensor.index, i);
//         shape.device = devices[i].clone();
//         shape.attr.insert("dtype".into(), AttrValue::new().apply(|x| x.set_field_type(DataType::DT_INT32)));
//         let mut value = crate::proto::tensor::TensorProto::new();
//         let mut x = crate::proto::tensor_shape::TensorShapeProto::new();
//         let mut dim = crate::proto::tensor_shape::TensorShapeProto_Dim::new();
//         dim.size = 1;
//         x.dim.push(dim);
//         value.dtype = DataType::DT_INT32;
//         value.tensor_shape = protobuf::SingularPtrField::some(x);
//         value.int_val.push(-1);
//         shape.attr.insert("value".into(), AttrValue::new().apply(|x| x.set_tensor(value)));
//         target.pb.node.push(shape);

//         let mut flat = tensor.node().make_node("Reshape".to_string());
//         flat.name += &format!("/ring_{}/aux_flat_{}", tensor.index, i);
//         flat.device = devices[i].clone();
//         flat.attr.insert("T".into(), dtype.clone());
//         flat.input.push(list[i].1.clone());
//         flat.input.push(format!("{}/ring_{}/aux_flat_{}/shape", basename, tensor.index, i));
//         set_input_size(&mut flat, 0, psize);
//         target.pb.node.push(flat);
//         format!("{}/ring_{}/aux_flat_{}", basename, tensor.index, i)
//     }).collect();

//     // 3. chunking
//     let mut chunks: Vec<Vec<String>> = (0..n).map(|i| {
//         let mut dim = tensor.node().make_node("Const".to_string());
//         dim.name += &format!("/ring_{}/aux_split_{}/split_dim", tensor.index, i);
//         dim.device = devices[i].clone();
//         dim.attr.insert("dtype".into(), AttrValue::new().apply(|x| x.set_field_type(DataType::DT_INT32)));
//         let mut value = crate::proto::tensor::TensorProto::new();
//         let shape = crate::proto::tensor_shape::TensorShapeProto::new();
//         value.dtype = DataType::DT_INT32;
//         value.tensor_shape = protobuf::SingularPtrField::some(shape);
//         value.int_val.push(0);
//         dim.attr.insert("value".into(), AttrValue::new().apply(|x| x.set_tensor(value)));
//         target.pb.node.push(dim);

//         let mut split = tensor.node().make_node("Split".to_string());
//         split.name += &format!("/ring_{}/aux_split_{}", tensor.index, i);
//         split.device = devices[i].clone();
//         split.input.push(format!("{}/ring_{}/aux_split_{}/split_dim", basename, tensor.index, i));
//         split.input.push(flats[i].clone());
//         split.attr.insert("T".into(), dtype.clone());
//         split.attr.insert("num_split".into(), AttrValue::new().apply(|x| x.set_i(n.try_into().unwrap())));
//         set_input_size(&mut split, 1, psize);
//         target.pb.node.push(split);

//         (0..n).map(|j| {
//             format!("{}/ring_{}/aux_split_{}:{}", basename, tensor.index, i, j)
//         }).collect()
//     }).collect();

//     // 4. n-1 rounds of reducing. the last modified chunks (i+n-2) have the full content
//     for round in 0..n-1 {
//         // at the r round, the r+i chunk on i node is replaced by the sum of r+i and r+i+1
//         for i in 0..n {
//             let mut add = tensor.node().make_node("Add".to_string());
//             add.name += &format!("/ring_{}/aux_add_{}_{}", tensor.index, i, round);
//             add.device = devices[i].clone();
//             add.input.push(chunks[i][(round+i) % n].clone());
//             add.input.push(chunks[(i+1) % n][(round+i) % n].clone());
//             add.attr.insert("T".into(), dtype.clone());
//             set_input_size(&mut add, 0, psize);
//             set_input_size(&mut add, 1, psize);
//             chunks[i][(round+i) % n] = add.name.clone();
//             target.pb.node.push(add);
//         }
//     }

//     // 5. n-1 rounds of gathering
//     for round in 0..n-1 {
//         for i in 0..n {
//             let mut identity = tensor.node().make_node("Identity".to_string());
//             identity.name += &format!("/ring_{}/aux_identity_{}_{}", tensor.index, i, round);
//             identity.device = devices[i].clone();
//             identity.attr.insert("T".into(), dtype.clone());
//             identity.input.push(chunks[(i+1) % n][(i+round+n-1) % n].clone());
//             set_input_size(&mut identity, 0, psize);
//             chunks[i][(i+round+n-1) % n] = identity.name.clone();
//             target.pb.node.push(identity);
//         }
//     }

//     // 6. concating
//     let concated: Vec<_> = chunks.into_iter().enumerate().map(|(i, chunk)| {
//         let mut axis = tensor.node().make_node("Const".to_string());
//         axis.name += &format!("/ring_{}/aux_concat_{}/axis", tensor.index, i);
//         axis.device = devices[i].clone();
//         axis.attr.insert("dtype".into(), AttrValue::new().apply(|x| x.set_field_type(DataType::DT_INT32)));
//         let mut value = crate::proto::tensor::TensorProto::new();
//         let shape = crate::proto::tensor_shape::TensorShapeProto::new();
//         value.dtype = DataType::DT_INT32;
//         value.tensor_shape = protobuf::SingularPtrField::some(shape);
//         value.int_val.push(0);
//         axis.attr.insert("value".into(), AttrValue::new().apply(|x| x.set_tensor(value)));
//         target.pb.node.push(axis);

//         let len = chunk.len(); // save it here since we will destruct it later
//         let mut concat = tensor.node().make_node("ConcatV2".to_string());
//         concat.name += &format!("/ring_{}/aux_concat_{}", tensor.index, i);
//         concat.device = devices[i].clone();
//         concat.input = chunk.into_iter().collect();
//         concat.input.push(format!("{}/ring_{}/aux_concat_{}/axis", basename, tensor.index, i));
//         concat.attr.insert("N".into(), AttrValue::new().apply(|x| x.set_i(n.try_into().unwrap())));
//         concat.attr.insert("T".into(), dtype.clone());
//         concat.attr.insert("Tidx".into(), AttrValue::new().apply(|x| x.set_field_type(DataType::DT_INT32)));
//         for j in 0..len {
//             set_input_size(&mut concat, j, psize);
//         }
//         target.pb.node.push(concat);

//         format!("{}/ring_{}/aux_concat_{}", basename, tensor.index, i)
//     }).collect();

//     // 7. restore shapes
//     concated.into_iter().zip(shapes).enumerate().map(|(i, (concat, shape))| {
//         let mut reshape = tensor.node().make_node("Reshape".to_string());
//         reshape.name += &format!("/ring_{}/aux_reshape_{}", tensor.index, i);
//         reshape.device = devices[i].clone();
//         reshape.attr.insert("T".into(), dtype.clone());
//         reshape.input.push(concat);
//         reshape.input.push(shape);
//         set_input_size(&mut reshape, 0, psize);
//         target.pb.node.push(reshape);
//         format!("{}/ring_{}/aux_reshape_{}", basename, tensor.index, i)
//     }).collect()
// }
