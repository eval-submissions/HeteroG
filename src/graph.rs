use oh_my_rust::*;
use protobuf::Message;
use crate::proto::{graph::GraphDef, node_def::NodeDef, attr_value::AttrValue, types::DataType};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;
use std::convert::TryInto;
use std::iter::FromIterator;
use crate::proto::attr_value::AttrValue_ListValue;
use std::hint::unreachable_unchecked;
use std::rc::Rc;
use std::cell::RefCell;
use std::hash::Hash;
use crate::misc::Target;

#[derive(Default)]
pub struct CollectiveState {
    pub groups: BTreeMap<Vec<usize>, u64>, // devices => group_key
    pub instances: Vec<Vec<usize>> // each instance is a list of index of Collective nodes in the target
}

impl CollectiveState {
    pub fn new_instance(&mut self) -> (&mut Vec<usize>, usize) {
        let key = self.instances.len();
        self.instances.push(vec![]);
        (self.instances.last_mut().unwrap(), key)
    }

    pub fn get_group(&mut self, devices: &[usize]) -> u64 {
        if let Some(key) = self.groups.get(devices) {
            *key
        } else {
            let key = self.groups.len() as _;
            self.groups.insert(devices.to_vec(), key);
            key
        }
    }
}

#[derive(Default)]
pub struct Graph {
    pub nodes: Vec<Node>, // This vector is partial ordered: inputs are guaranteed to appear earlier than descendants
    pub options: BTreeMap<String, String>,
    pub name_dict: BTreeMap<String, usize>,

    collective_state: CollectiveState
}

impl Graph {
    pub fn new(nodes: &[NodeDef]) -> Box<Self> {
        task!("building graph of {} nodes...", nodes.len());

        let mut g = Box::new(Graph { nodes: Vec::with_capacity(nodes.len()), ..Default::default() });

        // not always optimal, but good enough since the input is actually mostly ordered
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

        g.analyze();

        g
    }

    /// setup the replicas and links. Note that auxiliary nodes are already there by strategies.
    pub fn compile(&mut self, target: &mut Target) {
        task!("compiling graph of {} nodes...", self.nodes.len());
        for node in self.nodes.iter_mut() {
            node.compile(target)
        }

        self.add_control_dependencies_for_collective_nodes(target)
    }

    /// set flags and assign groups for nodes
    /// 1. mark tensors that has batchsize dimension with hand-crafted whitelist rules
    /// 2. group the nodes so that a.) all nodes inside a group is splittable and b.) all cross-group tensors are splittable
    /// 3. if all nodes in a group are replicated, use split, otherwise all replications are cache.
    fn analyze(&mut self) {
        // mark descendants of input
        let mut descendants_of_input: BTreeSet<usize> = BTreeSet::new();
        for (id, node) in self.nodes.iter_mut().enumerate() {
            for (input_id, index, _) in node.inputs.iter() {
                let input = &mut node.graph().nodes[*input_id];
                if descendants_of_input.contains(input_id) || input.is_input() {
                    input.get_output(*index).set_flag(Tensor::IS_FROM_INPUT);
                    descendants_of_input.insert(id);
                }
            }
        }

        // mark batch splittability
        for node in self.nodes.iter_mut() {
            match &node.raw_node.op[..] {
                "Placeholder" | "IteratorGetNext" | "Conv2D" | "MaxPool" | "MatMul" | "Conv2DBackpropInput" | "BiasAdd" => {
                    node.get_output(0).set_flag(Tensor::IS_BATCHED);
                },
                "Cast" | "ZerosLike" |"GreaterEqual" | "Neg" | "Log1p" | "Exp" | "Slice" |
                "Squeeze" | "Identity" | "Sigmoid" | "LeakyRelu" | "Relu" | "Tanh" => {
                    let (id, index, _) = &node.inputs[0];
                    if node.graph().nodes[*id].get_output(*index).has_flag(Tensor::IS_BATCHED) {
                        node.get_output(0).set_flag(Tensor::IS_BATCHED);
                    }
                },
                "Add" | "Sub" | "Mul" => for input_index in 0..=1 {
                    let (id, index, _) = &node.inputs[input_index];
                    if node.graph().nodes[*id].get_output(*index).has_flag(Tensor::IS_BATCHED) {
                        node.get_output(0).set_flag(Tensor::IS_BATCHED);
                        break
                    }
                },
                _ => {}
                // todo: Select?
                // todo: matmul has an attr that transpose the input on the fly
                // todo: shape -> fill or shape -> broadcast also gives a splittable tensor
            }
        }

        // hacks
        for (node_id, node) in self.nodes.iter_mut().enumerate() {
            match &node.raw_node.op[..] {
                "ApplyGradientDescent" => { // ensure gradients don't have batch dimension so they will be summed
                    let (id, index, _) = &node.inputs[2];
                    node.graph().nodes[*id].get_output(*index).unset_flag(Tensor::IS_BATCHED);
                    // assign it with the variable to the same group
                    let (id, _, _) = &node.inputs[0];
                    node.group = Some(Rc::new(RefCell::new(vec![node_id, *id])));
                    node.graph().nodes[*id].group = node.group.clone()
                },
                "ApplyAdam" => {
                    let (id, index, _) = &node.inputs[9];
                    node.graph().nodes[*id].get_output(*index).unset_flag(Tensor::IS_BATCHED);
                    // assign it with the variable and optimizer states to the same group
                    let (var_id, _, _) = &node.inputs[0];
                    let (m_id, _, _) = &node.inputs[1];
                    let (v_id, _, _) = &node.inputs[2];
                    node.group = Some(Rc::new(RefCell::new(vec![node_id, *var_id, *m_id, *v_id])));
                    node.graph().nodes[*var_id].group = node.group.clone();
                    node.graph().nodes[*m_id].group = node.group.clone();
                    node.graph().nodes[*v_id].group = node.group.clone()
                },
                "ScatterSub" => { // these tensors, however, should be concated
                    let (indices_id, indices_index, _) = &node.inputs[1];
                    let (updates_id, updates_index, _) = &node.inputs[2];
                    node.graph().nodes[*indices_id].get_output(*indices_index).set_flag(Tensor::IS_BATCHED);
                    node.graph().nodes[*updates_id].get_output(*updates_index).set_flag(Tensor::IS_BATCHED);
                    // assign it with the variable to the same group
                    let (id, _, _) = &node.inputs[0];
                    node.group = Some(Rc::new(RefCell::new(vec![node_id, *id])));
                    node.graph().nodes[*id].group = node.group.clone()
                },
                _ => {}
            }
        }

        // grouping
        for (node_id, node) in self.nodes.iter_mut().enumerate() {
            if !(node.is_input() || descendants_of_input.contains(&node_id)) { // if it is not a descendant of input, then it does not belong to any group
                continue
            }

            for (input_id, index, _) in node.inputs.iter() {
                let input = &mut node.graph().nodes[*input_id];
                if input.group.is_some() && !input.get_output(*index).has_flag(Tensor::IS_BATCHED) { // should be attached into the same group
                    let input_group = input.group.as_ref().cloned().unwrap();
                    match &node.group {
                        None => { // this node is not yet assigned into a group, so we just add it into the group of the input
                            node.group = Some(input_group.clone());
                            input_group.borrow_mut().push(node_id);
                        }
                        // this node already belongs to a group that is different from the one of the input. We merge the input group into the current group
                        Some(group) if &**group as *const _ != &*input_group as *const _ => {
                            for i in input_group.borrow().iter() {
                                node.graph().nodes[*i].group = Some(group.clone());
                                group.borrow_mut().push(*i);
                            }
                        }
                        Some(_) => {} // this node already has the same group with the input. Nothing to do here.
                    }
                }
            }

            if node.group.is_none() { // no constraint, assign a new group
                node.group = Some(Rc::new(RefCell::new(vec![node_id])));
            }
        }
    }

    pub fn get_groups(&self) -> BTreeMap<&str, Option<impl Hash + Ord>> {
        self.nodes.iter().map(|node| {
            (&node.raw_node.name[..], node.group.as_ref().map(|x| x.as_ptr()))
        }).collect()
    }

    fn add_control_dependencies_for_collective_nodes(&mut self, target: &mut Target) {
        // TODO: findout existing dependencies (added by fusing iterations) and avoid dead lock
        for pair in self.collective_state.instances.windows(2) {
            for &i in &pair[0] {
                for &j in &pair[1] {
                    let input = format!("^{}", target.pb.node.get(j).unwrap().name);
                    target.pb.node.get_mut(i).unwrap().input.push(input)
                }
            }
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd, Copy)]
pub enum FormKind { Full, Part }

#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd)]
pub struct Form {
    pub kind: FormKind,
    pub devices: Vec<usize> // The Vec must be sorted and not empty, but may contains repeated elements (put multiple replicas on the same device)
}

impl Form {
    pub fn is_full(&self) -> bool {
        self.kind == FormKind::Full
    }

    pub fn is_part(&self) -> bool {
        self.kind == FormKind::Part
    }

    pub fn ndev(&self) -> usize {
        self.devices.len()
    }

    // TODO: use to_string() and parse()?
    pub fn code(&self) -> String {
        let mut x = String::from(if self.is_full() {"full"} else {"part"});
        for d in self.devices.iter() {
            x += "_";
            x += &d.to_string();
        }
        x
    }

    pub fn from_code(code: &str) -> Self {
        let segs: Vec<_> = code.split('_').collect();
        let kind = match segs[0] {
            "full" => FormKind::Full,
            "part" => FormKind::Part,
            _ => unreachable!()
        };
        Self { kind, devices: segs[1..].iter().map(|x| x.parse().unwrap()).collect() }
    }

    pub fn valid(&self) -> bool {
        !self.devices.is_empty()
    }
}

type Group = Rc<RefCell<Vec<usize>>>;

pub struct Node {
    pub graph: *const Graph,
    pub raw_node: NodeDef,
    pub controls: Vec<usize>, // TODO: more consideration for control dependencies that added aux nodes
    pub inputs: Vec<(usize, usize, FormKind)>, // nodeid, index, formkind (defaults to full)
    pub outputs: Vec<Tensor>,
    pub form: Form, // the form of the node, which is also a tensor form for all its outputs
    pub group: Option<Group>,
}

impl Node {
    pub fn new(graph: &Graph, raw_node: NodeDef) -> Self {
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
            group: None
        }
    }

    #[allow(clippy::mut_from_ref)]
    pub fn graph<'a>(&self) -> &'a mut Graph {
        unsafe { &mut *(self.graph as *mut Graph) }
    }

    pub fn get_output(&mut self, index: usize) -> &mut Tensor {
        while self.outputs.len() <= index {
            self.outputs.push(Tensor::new(self, self.outputs.len()))
        }

        &mut self.outputs[index]
    }

    pub fn replicated(&self) -> Option<bool> {
        match self.form.ndev() {
            0 => None,
            1 => Some(false),
            _ => Some(true)
        }
    }

    /// add an edited node into the target. Requires all inputs to be compiled first
    fn compile(&mut self, target: &mut Target) {
        if self.graph().options.get("log_forms").map(|x| x == "True").unwrap_or(false) {
            info!("compile: {} {:?} {:?}", self.raw_node.name, self.form, self.inputs);
        }

        for (replica_index, device_id) in self.form.devices.iter().enumerate() {
            // 0. replace placeholders
            if self.raw_node.op == "Placeholder" {
                if let Some(batchsize) = self.graph().options.get("replace_placeholder") {
                    let batchsize: usize = batchsize.parse().unwrap();
                    let mut shape: Vec<Option<usize>> = self.raw_node.attr["_output_shapes"].get_list().shape[0].dim.iter().map(|x| x.size.try_into().ok()).collect();
                    if self.form.is_part() {
                        shape[0].replace(batchsize / self.form.ndev() as usize);
                    } else {
                        shape[0].replace(batchsize);
                    }

                    let mut shape_node = self.make_node("Const".to_string());
                    shape_node.name += &format!("/aux_replace_placeholder_{}/shape", replica_index);
                    shape_node.device = target.devices[*device_id].clone();
                    shape_node.attr.insert("dtype".into(), AttrValue::new().apply(|x| x.set_field_type(DataType::DT_INT32)));
                    let value = crate::proto::tensor::TensorProto::new().apply(|x| {
                        x.set_dtype(DataType::DT_INT32);
                        x.set_tensor_shape(crate::proto::tensor_shape::TensorShapeProto::new().apply(|s| s.set_dim([shape.len()].iter().map(|x| {crate::proto::tensor_shape::TensorShapeProto_Dim::new().apply(|fuck| fuck.size = *x as _)}).collect())));
                        for dim in shape {
                            x.int_val.push(dim.unwrap() as _);
                        }
                    });
                    shape_node.attr.insert("value".into(), AttrValue::new().apply(|x| x.set_tensor(value)));

                    let mut random_node = NodeDef::new().apply(|x| x.op = "RandomUniform".to_string());
                    random_node.name = self.replica(replica_index);
                    random_node.device = target.devices[*device_id].clone();
                    set_origin(&mut random_node, &self.raw_node.name);
                    set_form(&mut random_node, &self.form.code());
                    random_node.input.push(shape_node.name.clone());
                    random_node.attr.insert("T".into(), AttrValue::new().apply(|x| x.set_field_type(DataType::DT_INT32)));
                    random_node.attr.insert("dtype".into(), get_dtype(&self.raw_node, 0));

                    target.pb.node.push(shape_node);
                    target.pb.node.push(random_node);
                    continue
                }
            }

            // 1. setup basic node info
            let mut node = self.raw_node.clone();
            node.name = self.replica(replica_index);
            node.device = target.devices[*device_id].clone();
            set_origin(&mut node, &self.raw_node.name);
            set_form(&mut node, &self.form.code());

            // 2. link inputs and set size
            node.input = self.inputs.iter().copied().enumerate().map(|(i, (node_id, index, kind))| {
                let input_tensor = &mut self.graph().nodes[node_id].get_output(index);
                set_input_size(&mut node, i, match kind {
                    FormKind::Full => input_tensor.get_size(),
                    FormKind::Part => input_tensor.get_size() / self.form.ndev() as u64,
                });
                let input_names = input_tensor.as_form(&Form { kind, devices: self.form.devices.clone() }, target);
                input_names[replica_index].clone()
            }).collect();

            // 3. add control dependencies
            if self.raw_node.op == "NoOp" { // TODO: what's the consequence?
                for node_id in self.controls.iter() {
                    let dep_node = &self.graph().nodes[*node_id];

                    for i in 0..dep_node.form.ndev() {
                        node.input.push(format!("^{}", dep_node.replica(i)))
                    }
                }
            }

            target.pb.node.push(node)
        }
    }

    fn replica(&self, index: usize) -> String { // TODO: should this method exist?
        format!("{}/replica_{}", self.raw_node.name, index)
    }

    pub fn is_input(&self) -> bool {
        self.raw_node.op == "Placeholder" || self.raw_node.op == "IteratorGetNext"
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

pub struct Tensor {
    pub node: *const Node,
    pub index: usize,
    pub forms: BTreeMap<Form, Box<[String]>>,
    pub flags: u8, // flags indicate the types and roles of a tensor. It affects how the tensor is treated when changing forms
}

impl Tensor {
    pub const IS_FROM_INPUT: u8 = 0x01; // this tensor is a descendant of an input node.
    pub const IS_BATCHED: u8 = 0x02; // this tensor's first dimension is batch size.
    pub const IS_SHAPE: u8 = 0x04; // currently not used
    pub const IS_FIXED: u8 = 0x80; // this tensor's form is provided by strategy and should not be altered

    pub fn new(node: &Node, index: usize) -> Self {
        Tensor { node, index, forms: BTreeMap::new(), flags: 0 }
    }

    pub fn original_name(&self) -> String {
        if self.index == 0 {
            self.node().raw_node.name.clone()
        } else {
            format!("{}:{}", self.node().raw_node.name, self.index)
        }
    }

    pub fn node<'a>(&self) -> &'a Node {
        unsafe { &*self.node }
    }

    pub fn get_shape(&self) -> Vec<usize> {
        // sucks: the output shape of BroadcastGradientArgs is always unknown even if inputs are fixed
        // and ops like `Sum` (requires the dimension to sum along with) and `Fill` operates differently with different inputs
        let mut shape: Vec<_> = self.node().raw_node.attr["_output_shapes"].get_list().shape[self.index].dim.iter().map(|x| x.size.try_into().ok()).collect();
        if let Some(batchsize) = self.node().graph().options.get("fill_batchsize") {
            // if self.has_flag(Self::IS_BATCHED) && self.has_flag(Self::IS_FROM_INPUT) && shape[0].is_none() && shape[1..].iter().all(|x| x.is_some()) {
            if !shape.is_empty() && shape[0].is_none() { // relax the requirement
                shape[0] = Some(batchsize.parse().unwrap());
            }
        }
        shape.into_iter().collect::<Option<_>>().unwrap_or_else(Vec::new)
    }

    pub fn get_size(&self) -> u64 {
        #[allow(clippy::unnecessary_fold)]
        (self.get_shape().iter().fold(1, |x, y| x * y) * 4).try_into().unwrap()
    }

    pub fn has_flag(&self, flag: u8) -> bool {
        self.flags & flag != 0
    }

    pub fn set_flag(&mut self, flag: u8) {
        self.flags |= flag
    }

    pub fn unset_flag(&mut self, flag: u8) {
        self.flags &= !flag
    }

    // get the names as the specified form
    pub fn as_form(&mut self, form: &Form, target: &mut Target) -> &[String] {
        if !self.forms.contains_key(form) {
            if self.has_flag(Self::IS_FIXED) {
                panic!("BUG: no form {:?} provided for {}", form, self.original_name())
            }

            let names = if form == &self.node().form {
                (0..form.ndev()).map(|i| format!("{}:{}", self.node().replica(i), self.index)).collect()
            } else {
                let node_kind = self.node().form.kind;
                match (form.kind, node_kind) {
                    (FormKind::Full, FormKind::Full) => self.replicate_broadcast(&self.node().form, form, target),
                    (FormKind::Part, FormKind::Full) => {
                        if self.has_flag(Self::IS_SHAPE) {
                            unimplemented!()
                        }

//                        if self.has_flag(Self::IS_BATCHED) {
                            self.replicate_split(&self.node().form, form, target)
//                        } else {
//                            panic!("cannot split a unbatched tensor")
//                        }
                    },
                    (FormKind::Full, FormKind::Part) => {
                        if self.has_flag(Self::IS_SHAPE) {
                            unimplemented!()
                        }

                        if self.has_flag(Self::IS_BATCHED) {
                            self.aggregate_cat(&self.node().form, form, target)
                        } else { // if it is not batched but is split, the only possibility is it is inherited from a split parent, so it must be a gradient
                            self.aggregate_sum(&self.node().form, form, target)
                        }
                    },
                    (FormKind::Part, FormKind::Part) => {
                        if self.has_flag(Self::IS_SHAPE) {
                            unimplemented!()
                        }

                        // if self.has_flag(Self::IS_BATCHED) {
                        //     self.resplit(&self.node().form, form, target)
                        // } else {
                            // unimplemented!("cannot resplit a unbatched tensor")
                            // there is currently a hack in resplit that copy parts if the number matches. Move this logic out and mark this case a bug.
                            self.resplit(&self.node().form, form, target)
                        // }
                    },
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
        assert!(from.valid() && to.valid() && from.is_part() && to.is_full());

        let mut addn = self.node().make_node("AddN".to_string());
        addn.name += &format!("/{}_{}/aux_sum", self.index, to.code());
        addn.device = target.devices[to.devices[0]].clone();
        addn.attr.insert("N".into(), AttrValue::new().apply(|x| x.set_i(from.ndev() as _)));
        addn.attr.insert("T".into(), get_dtype(&self.node().raw_node, self.index));
        addn.input = self.as_form(from, target).iter().cloned().collect();
        for i in 0..from.ndev() {
            set_input_size(&mut addn, i, self.get_size() / from.ndev() as u64)
        }

        let result = vec![addn.name.clone(); to.ndev()].into_boxed_slice();
        target.pb.node.push(addn);
        result
    }

    // TODO: share the same axis nodes for all concating (and do the same thing for dim nodes in splitting)
    pub fn aggregate_cat(&mut self, from: &Form, to: &Form, target: &mut Target) -> Box<[String]> {
        assert!(from.valid() && to.valid() && from.is_part() && to.is_full());

        let mut axis = self.node().make_node("Const".to_string());
        axis.name += &format!("/{}_{}/aux_concat/axis", self.index, to.code());
        axis.device = target.devices[to.devices[0]].clone();
        axis.attr.insert("dtype".into(), AttrValue::new().apply(|x| x.set_field_type(DataType::DT_INT32)));
        let value = crate::proto::tensor::TensorProto::new().apply(|x| {
            x.set_dtype(DataType::DT_INT32);
            x.set_tensor_shape(crate::proto::tensor_shape::TensorShapeProto::new());
            x.int_val.push(0);
        });
        axis.attr.insert("value".into(), AttrValue::new().apply(|x| x.set_tensor(value)));

        let mut concat = self.node().make_node("ConcatV2".to_string());
        concat.name += &format!("/{}_{}/aux_concat/concat", self.index, to.code());
        concat.device = target.devices[to.devices[0]].clone();
        concat.input = self.as_form(from, target).iter().cloned().collect();
        concat.input.push(axis.name.clone());
        concat.attr.insert("N".into(), AttrValue::new().apply(|x| x.set_i(from.ndev() as _)));
        concat.attr.insert("T".into(), get_dtype(&self.node().raw_node, self.index));
        concat.attr.insert("Tidx".into(), AttrValue::new().apply(|x| x.set_field_type(DataType::DT_INT32)));
        for i in 0..from.ndev() {
            set_input_size(&mut concat, i, self.get_size() / from.ndev() as u64)
        }

        let result = vec![concat.name.clone(); to.ndev()].into_boxed_slice();
        target.pb.node.push(axis);
        target.pb.node.push(concat);
        result
    }

    pub fn replicate_broadcast(&mut self, from: &Form, to: &Form, target: &mut Target) -> Box<[String]> {
        assert!(from.valid() && to.valid() && from.is_full() && to.is_full());

        let raw = self.as_form(&self.node().form, target).to_vec(); // TODO: no clone?
        to.devices.iter().map(|device_id| {
            from.devices.iter().position(|x| *x == *device_id).map(|ind| raw[ind].clone()).unwrap_or_else(|| raw[0].clone())
        }).collect()
    }

    // currenly we only split from the first replica. Future we can split on every device and use the local copy to reduce transfering
    pub fn replicate_split(&mut self, from: &Form, to: &Form, target: &mut Target) -> Box<[String]> {
        assert!(from.valid() && to.valid() && from.is_full() && to.is_part());

        let mut dim = self.node().make_node("Const".to_string());
        dim.name += &format!("/{}_{}/aux_split/dim", self.index, to.code());
        dim.device = target.devices[from.devices[0]].clone();
        dim.attr.insert("dtype".into(), AttrValue::new().apply(|x| x.set_field_type(DataType::DT_INT32)));
        let value = crate::proto::tensor::TensorProto::new().apply(|x| {
            x.set_dtype(DataType::DT_INT32);
            x.set_tensor_shape(crate::proto::tensor_shape::TensorShapeProto::new());
            x.int_val.push(0);
        });
        dim.attr.insert("value".into(), AttrValue::new().apply(|x| x.set_tensor(value)));

        let mut split = self.node().make_node("Split".to_string());
        split.name += &format!("/{}_{}/aux_split/split", self.index, to.code());
        split.device = target.devices[from.devices[0]].clone();
        split.input.push(dim.name.clone());
        split.input.push(self.as_form(from, target)[0].clone());
        split.attr.insert("T".into(), get_dtype(&self.node().raw_node, self.index));
        split.attr.insert("num_split".into(), AttrValue::new().apply(|x| x.set_i(to.ndev() as _)));
        set_input_size(&mut split, 1, self.get_size());

        let result = (0..to.ndev()).map(|i| format!("{}:{}", split.name, i)).collect();
        target.pb.node.push(dim);
        target.pb.node.push(split);
        result
    }

    pub fn resplit(&mut self, from: &Form, to: &Form, target: &mut Target) -> Box<[String]> {
        assert!(from.valid() && to.valid() && from.is_part() && to.is_part());

        if from.ndev() == to.ndev() { // special case: if the number are the same, just forward. TODO: use replicas on the same device when possible
            return self.as_form(from, target).to_vec().into_boxed_slice()
        }

        let gcd = { // the number of intermediat concated nodes
            let mut a = from.ndev();
            let mut b = to.ndev();
            while a != b {
                if a > b {
                    a -= b;
                } else {
                    b -= a;
                }
            }
            a
        };

        self.as_form(from, target).to_vec().chunks(from.ndev() / gcd).enumerate().map(|(i, chunk)| {
            let dest = from.devices[i * chunk.len()];

            // special case: no need to concat
            if chunk.len() == 1 {
                return (dest, chunk[0].clone())
            }

            let mut axis = self.node().make_node("Const".to_string());
            axis.name += &format!("/{}_{}/aux_resplit_{}/concat_axis", self.index, to.code(), i);
            axis.device = target.devices[dest].clone();
            axis.attr.insert("dtype".into(), AttrValue::new().apply(|x| x.set_field_type(DataType::DT_INT32)));
            let value = crate::proto::tensor::TensorProto::new().apply(|x| {
                x.set_dtype(DataType::DT_INT32);
                x.set_tensor_shape(crate::proto::tensor_shape::TensorShapeProto::new());
                x.int_val.push(0);
            });
            axis.attr.insert("value".into(), AttrValue::new().apply(|x| x.set_tensor(value)));

            let mut concat = self.node().make_node("ConcatV2".to_string());
            concat.name += &format!("/{}_{}/aux_resplit_{}/concat", self.index, to.code(), i);
            concat.device = target.devices[dest].clone();
            concat.input = chunk.iter().cloned().collect();
            concat.input.push(axis.name.clone());
            concat.attr.insert("N".into(), AttrValue::new().apply(|x| x.set_i(chunk.len() as _)));
            concat.attr.insert("T".into(), get_dtype(&self.node().raw_node, self.index));
            concat.attr.insert("Tidx".into(), AttrValue::new().apply(|x| x.set_field_type(DataType::DT_INT32)));
            for j in 0..chunk.len() {
                set_input_size(&mut concat, j, self.get_size() / from.ndev() as u64)
            }

            let result = concat.name.clone();
            target.pb.node.push(axis);
            target.pb.node.push(concat);
            (dest, result)
        }).collect::<Vec<_>>().iter().zip(to.devices.chunks(to.ndev() / gcd)).enumerate().flat_map(|(i, ((concat_place, concated), devices))| {
            if devices.len() == 1 { // special case: no need to split
                return vec![concated.clone()] // TODO: use another return type for this closure? Ideally do not collect and return a dyn IntoIterator instead
            }

            let mut dim = self.node().make_node("Const".to_string());
            dim.name += &format!("/{}_{}/aux_resplit_{}/split_dim", self.index, to.code(), i);
            dim.device = target.devices[*concat_place].clone();
            dim.attr.insert("dtype".into(), AttrValue::new().apply(|x| x.set_field_type(DataType::DT_INT32)));
            let value = crate::proto::tensor::TensorProto::new().apply(|x| {
                x.set_dtype(DataType::DT_INT32);
                x.set_tensor_shape(crate::proto::tensor_shape::TensorShapeProto::new());
                x.int_val.push(0);
            });
            dim.attr.insert("value".into(), AttrValue::new().apply(|x| x.set_tensor(value)));

            let mut split = self.node().make_node("Split".to_string());
            split.name += &format!("/{}_{}/aux_resplit_{}/split", self.index, to.code(), i);
            split.device = target.devices[*concat_place].clone();
            split.input.push(dim.name.clone());
            split.input.push(concated.clone());
            split.attr.insert("T".into(), get_dtype(&self.node().raw_node, self.index));
            split.attr.insert("num_split".into(), AttrValue::new().apply(|x| x.set_i(devices.len() as _)));
            set_input_size(&mut split, 1, self.get_size() / gcd as u64);

            let result = (0..to.ndev() / gcd).map({
                let name = split.name.clone();
                move |i| format!("{}:{}", name, i)
            }).collect();
            target.pb.node.push(dim);
            target.pb.node.push(split);
            result
        }).collect()
    }

    pub fn all_reduce_sum_nccl(&mut self, from: &Form, to: &Form, target: &mut Target) -> Box<[String]> {
        // to all_sum n tensors (can be on the same device), one should have n NcclAllReduce nodes with the same shared_name attr
        // each node have only *one* input, and should be on the same device of the input. The output of these nodes will be the same

        assert!(target.devices.windows(2).all(|w| task_name(&w[0]) == task_name(&w[1]))); // This nodes only works intra-task
        assert!(from.valid() && to.valid() && from.is_part() && to.is_full() && from.devices == to.devices);

        let index = self.index;

        for (i, device_id) in from.devices.iter().enumerate() {
            let mut nccl = self.node().make_node("NcclAllReduce".to_string());
            nccl.name += &format!("/{}_{}/aux_nccl_{}", index, to.code(), i);
            nccl.device = target.devices[*device_id].clone();
            nccl.attr.insert("reduction".into(), AttrValue::new().apply(|x| x.set_s(b"sum".to_vec())));
            nccl.attr.insert("T".into(), get_dtype(&self.node().raw_node, self.index));
            nccl.attr.insert("num_devices".into(), AttrValue::new().apply(|x| x.set_i(from.ndev() as _)));
            nccl.attr.insert("shared_name".into(), AttrValue::new().apply(|x| x.set_s(self.original_name().into_bytes())));
            nccl.input.push(self.as_form(from, target)[i].clone());
            set_input_size(&mut nccl, 0, self.get_size() / from.ndev() as u64);

            target.pb.node.push(nccl)
        }

        (0..from.ndev()).map(|i| format!("{}/{}_{}/aux_nccl_{}", self.node().raw_node.name, self.index, to.code(), i)).collect()
    }

    pub fn all_reduce_sum_collective(&mut self, from: &Form, to: &Form, target: &mut Target) -> Box<[String]> {
        // each node have only *one* input, and should be on the same device of the input. The output of these nodes will be the same
        // group_key: not sure, guess is nccl group, so operations on *the same set of devices* could share the same group_key
        // instance_key: each operation use a unique instance_key, pretty much like "shared_name" in NcclAllReduce

        assert!(from.valid() && to.valid() && from.is_part() && to.is_full() && from.devices == to.devices);

        let part_size = self.get_size() / from.ndev() as u64;

        let mut local_groups: BTreeMap<usize, Vec<String>> = BTreeMap::new();
        for (i, device_id) in from.devices.iter().copied().enumerate() {
            let name = self.as_form(from, target)[i].clone();
            local_groups.entry(device_id).or_default().push(name)
        }

        let local_summed: BTreeMap<_, _> = local_groups.iter().map(|(device_id, local_nodes)| (*device_id, match local_nodes.len() {
            0 => unreachable!(),
            1 => local_nodes[0].clone(), // TODO: destruct the vector and take the element
            _ => {
                let mut addn = self.node().make_node("AddN".to_string());
                addn.name += &format!("/{}_{}_{}/aux_sum", self.index, to.code(), device_id);
                addn.device = target.devices[*device_id].clone();
                addn.attr.insert("N".into(), AttrValue::new().apply(|x| x.set_i(local_nodes.len() as _)));
                addn.attr.insert("T".into(), get_dtype(&self.node().raw_node, self.index));
                addn.input = local_nodes.iter().cloned().collect();
                for i in 0..local_nodes.len() {
                    set_input_size(&mut addn, i, part_size)
                }
                let name = addn.name.clone();
                target.pb.node.push(addn);
                name
            }
        })).collect();

        let state = &mut self.node().graph().collective_state;
        let group_key = state.get_group(&local_summed.keys().copied().collect::<Vec<_>>()); // it is sorted by BTreeMap
        let (instance, instance_key) = state.new_instance();

        let local_reduced: BTreeMap<_, _> = local_summed.iter().map(|(device_id, local_name)| {
            let mut node = self.node().make_node("CollectiveReduce".to_string());
            node.name += &format!("/{}_{}_{}/aux_collective", self.index, to.code(), device_id);
            node.device = target.devices[*device_id].clone();
            node.attr.insert("T".into(), get_dtype(&self.node().raw_node, self.index));
            node.attr.insert("final_op".into(), AttrValue::new().apply(|x| x.set_s(b"Id".to_vec())));
            node.attr.insert("merge_op".into(), AttrValue::new().apply(|x| x.set_s(b"Add".to_vec())));
            node.attr.insert("group_key".into(), AttrValue::new().apply(|x| x.set_i(group_key as _)));
            node.attr.insert("group_size".into(), AttrValue::new().apply(|x| x.set_i(local_summed.len() as _)));
            node.attr.insert("instance_key".into(), AttrValue::new().apply(|x| x.set_i(instance_key as _)));
            node.attr.insert("subdiv_offsets".into(), AttrValue::new().apply(|x| x.mut_list().i = vec![0]));
            node.input.push(local_name.clone());
            set_input_size(&mut node, 0, part_size);

            instance.push(target.pb.node.len());
            let name = node.name.clone();
            target.pb.node.push(node);
            (device_id, name)
        }).collect();

        from.devices.iter().map(|device_id| local_reduced[device_id].clone()).collect()
    }

    pub fn all_reduce_cat_collective(&mut self, from: &Form, to: &Form, target: &mut Target) -> Box<[String]> {
        assert!(from.valid() && to.valid() && from.is_part() && to.is_full() && from.devices == to.devices && BTreeSet::from_iter(from.devices.iter()).len() == from.devices.len());

        let part_size = self.get_size() / from.ndev() as u64;
        let state = &mut self.node().graph().collective_state;
        let group_key = state.get_group(&from.devices.clone().apply(|x| x.sort_unstable())); // sorted
        let (instance, instance_key) = state.new_instance();

        let list = self.as_form(from, target).to_vec();
        let local_reduced: BTreeMap<_, _> = from.devices.iter().zip(list.iter()).map(|(device_id, local_name)| {
            let mut node = self.node().make_node("CollectiveGather".to_string());
            node.name += &format!("/{}_{}_{}/aux_collective", self.index, to.code(), device_id);
            node.device = target.devices[*device_id].clone();
            node.attr.insert("T".into(), get_dtype(&self.node().raw_node, self.index));
            node.attr.insert("group_key".into(), AttrValue::new().apply(|x| x.set_i(group_key as _)));
            node.attr.insert("group_size".into(), AttrValue::new().apply(|x| x.set_i(from.devices.len() as _)));
            node.attr.insert("instance_key".into(), AttrValue::new().apply(|x| x.set_i(instance_key as _)));
            node.attr.insert("shape".into(), AttrValue::new().apply(|x| x.mut_shape().ignore()));
            node.input.push(local_name.clone());
            set_input_size(&mut node, 0, part_size);

            instance.push(target.pb.node.len());
            let name = node.name.clone();
            target.pb.node.push(node);
            (device_id, name)
        }).collect();

        from.devices.iter().map(|device_id| local_reduced[device_id].clone()).collect()
    }

    pub fn all_reduce_sum_ring(&mut self, from: &Form, to: &Form, target: &mut Target) -> Box<[String]> {
        assert!(from.valid() && to.valid() && from.is_part() && to.is_full() && from.devices == to.devices);

        let devices: Vec<_> = from.devices.iter().map(|id| target.devices[*id].clone()).collect();
        let n = devices.len();
        let dtype = get_dtype(&self.node().raw_node, self.index);
        let psize = self.get_size() / from.ndev() as u64;
        let list = self.as_form(from, target).to_vec();

        // 1. recording the shape
        let shapes: Vec<_> = (0..n).map(|i| {
            let mut shape = self.node().make_node("Shape".to_string());
            shape.name += &format!("/{}_{}/aux_ring/shape_{}", to.code(), self.index, i);
            shape.device = devices[i].clone();
            shape.attr.insert("T".into(), dtype.clone());
            shape.input.push(list[i].clone());
            set_input_size(&mut shape, 0, psize);
            let ret = shape.name.clone();
            target.pb.node.push(shape);
            ret
        }).collect();

        // 2. flattening
        let flats: Vec<_> = (0..n).map(|i| {
            let mut shape = self.node().make_node("Const".to_string());
            shape.name += &format!("/{}_{}/aux_ring/flat_{}/shape", to.code(), self.index, i);
            shape.device = devices[i].clone();
            shape.attr.insert("dtype".into(), AttrValue::new().apply(|x| x.set_field_type(DataType::DT_INT32)));
            let mut value = crate::proto::tensor::TensorProto::new();
            let mut x = crate::proto::tensor_shape::TensorShapeProto::new();
            let mut dim = crate::proto::tensor_shape::TensorShapeProto_Dim::new();
            dim.size = 1;
            x.dim.push(dim);
            value.dtype = DataType::DT_INT32;
            value.tensor_shape = protobuf::SingularPtrField::some(x);
            value.int_val.push(-1);
            shape.attr.insert("value".into(), AttrValue::new().apply(|x| x.set_tensor(value)));

            let mut flat = self.node().make_node("Reshape".to_string());
            flat.name += &format!("/{}_{}/aux_ring/flat_{}/flat", to.code(), self.index, i);
            flat.device = devices[i].clone();
            flat.attr.insert("T".into(), dtype.clone());
            flat.input.push(list[i].clone());
            flat.input.push(shape.name.clone());
            set_input_size(&mut flat, 0, psize);

            let ret = flat.name.clone();
            target.pb.node.push(shape);
            target.pb.node.push(flat);
            ret
        }).collect();

        // 3. chunking
        let mut chunks: Vec<Vec<String>> = (0..n).map(|i| {
            let mut dim = self.node().make_node("Const".to_string());
            dim.name += &format!("/{}_{}/aux_ring/split_{}/dim", to.code(), self.index, i);
            dim.device = devices[i].clone();
            dim.attr.insert("dtype".into(), AttrValue::new().apply(|x| x.set_field_type(DataType::DT_INT32)));
            let mut value = crate::proto::tensor::TensorProto::new();
            let shape = crate::proto::tensor_shape::TensorShapeProto::new();
            value.dtype = DataType::DT_INT32;
            value.tensor_shape = protobuf::SingularPtrField::some(shape);
            value.int_val.push(0);
            dim.attr.insert("value".into(), AttrValue::new().apply(|x| x.set_tensor(value)));

            let mut split = self.node().make_node("Split".to_string());
            split.name += &format!("/{}_{}/aux_ring/split_{}/split", to.code(), self.index, i);
            split.device = devices[i].clone();
            split.input.push(dim.name.clone());
            split.input.push(flats[i].clone());
            split.attr.insert("T".into(), dtype.clone());
            split.attr.insert("num_split".into(), AttrValue::new().apply(|x| x.set_i(n as _)));
            set_input_size(&mut split, 1, psize);

            let ret = split.name.clone();
            target.pb.node.push(dim);
            target.pb.node.push(split);

            (0..n).map(|x| format!("{}:{}", ret, x)).collect()
        }).collect();

        // 4. n-1 rounds of reducing. the last modified chunks (i+n-2) have the full content
        for round in 0..n-1 {
            // at the r round, the r+i chunk on i node is replaced by the sum of r+i and r+i+1
            for i in 0..n {
                let mut add = self.node().make_node("Add".to_string());
                add.name += &format!("/{}_{}/aux_ring/add_{}_{}", to.code(), self.index, i, round);
                add.device = devices[i].clone();
                add.input.push(chunks[i][(round+i) % n].clone());
                add.input.push(chunks[(i+1) % n][(round+i) % n].clone());
                add.attr.insert("T".into(), dtype.clone());
                set_input_size(&mut add, 0, psize);
                set_input_size(&mut add, 1, psize);
                chunks[i][(round+i) % n] = add.name.clone();
                target.pb.node.push(add);
            }
        }

        // 5. n-1 rounds of gathering
        for round in 0..n-1 {
            for i in 0..n {
                let mut identity = self.node().make_node("Identity".to_string());
                identity.name += &format!("/{}_{}/aux_ring/identity_{}_{}", to.code(), self.index, i, round);
                identity.device = devices[i].clone();
                identity.attr.insert("T".into(), dtype.clone());
                identity.input.push(chunks[(i+1) % n][(i+round+n-1) % n].clone());
                set_input_size(&mut identity, 0, psize);
                chunks[i][(i+round+n-1) % n] = identity.name.clone();
                target.pb.node.push(identity);
            }
        }

        // 6. concating
        let concated: Vec<_> = chunks.into_iter().enumerate().map(|(i, chunk)| {
            let mut axis = self.node().make_node("Const".to_string());
            axis.name += &format!("/{}_{}/aux_ring/concat_{}/axis", to.code(), self.index, i);
            axis.device = devices[i].clone();
            axis.attr.insert("dtype".into(), AttrValue::new().apply(|x| x.set_field_type(DataType::DT_INT32)));
            let mut value = crate::proto::tensor::TensorProto::new();
            let shape = crate::proto::tensor_shape::TensorShapeProto::new();
            value.dtype = DataType::DT_INT32;
            value.tensor_shape = protobuf::SingularPtrField::some(shape);
            value.int_val.push(0);
            axis.attr.insert("value".into(), AttrValue::new().apply(|x| x.set_tensor(value)));

            let len = chunk.len(); // save it here since we will destruct it later
            let mut concat = self.node().make_node("ConcatV2".to_string());
            concat.name += &format!("/{}_{}/aux_ring/concat_{}/concat", to.code(), self.index, i);
            concat.device = devices[i].clone();
            concat.input = chunk.into_iter().collect();
            concat.input.push(axis.name.clone());
            concat.attr.insert("N".into(), AttrValue::new().apply(|x| x.set_i(n as _)));
            concat.attr.insert("T".into(), dtype.clone());
            concat.attr.insert("Tidx".into(), AttrValue::new().apply(|x| x.set_field_type(DataType::DT_INT32)));
            for j in 0..len {
                set_input_size(&mut concat, j, psize);
            }

            let ret = concat.name.clone();
            target.pb.node.push(axis);
            target.pb.node.push(concat);
            ret
        }).collect();

        // 7. restore shapes
        concated.into_iter().zip(shapes).enumerate().map(|(i, (concat, shape))| {
            let mut reshape = self.node().make_node("Reshape".to_string());
            reshape.name += &format!("/{}_{}/aux_ring/reshape_{}", to.code(), self.index, i);
            reshape.device = devices[i].clone();
            reshape.attr.insert("T".into(), dtype.clone());
            reshape.input.push(concat);
            reshape.input.push(shape);
            set_input_size(&mut reshape, 0, psize);

            let ret = reshape.name.clone();
            target.pb.node.push(reshape);
            ret
        }).collect()
    }
}

fn set_origin(node: &mut NodeDef, origin: &str) {
    node.attr.insert("_tge_origin".to_string(), AttrValue::new().apply(|x| x.set_s(origin.as_bytes().to_vec())));
}

fn set_belong_to(node: &mut NodeDef, belong_to: &str) {
    node.attr.insert("_tge_belong_to".to_string(), AttrValue::new().apply(|x| x.set_s(belong_to.as_bytes().to_vec())));
}

fn set_input_size(node: &mut NodeDef, index: usize, size: u64) {
    let sizes = &mut node.attr.entry("_tge_input_sizes".to_string()).or_insert_with(AttrValue::new).mut_list().i;
    if sizes.len() <= index {
        sizes.resize(index+1, 0)
    }
    sizes[index] = size as _;
}

fn set_form(node: &mut NodeDef, form_code: &str) {
    node.attr.insert("_tge_form".to_string(), AttrValue::new().apply(|x| x.set_s(form_code.as_bytes().to_vec())));
}

// TODO: This function is currently a stub. Need to parse ops.pbtxt and follow type or type_attr.
fn get_dtype(x: &NodeDef, i: usize) -> AttrValue {
    match &x.op[..] {
        "Greater" | "GreaterEqual" => AttrValue::new().apply(|x| x.set_field_type(DataType::DT_BOOL)),
        "Shape" | "ShapeN" => x.attr.get("out_type").cloned().unwrap_or_else(|| AttrValue::new().apply(|x| x.set_field_type(DataType::DT_INT32))),
        "Cast" => x.attr.get("DstT").cloned().unwrap(),
        "IteratorGetNext" => AttrValue::new().apply(|v| v.set_field_type(x.attr.get("output_types").unwrap().get_list().get_field_type()[i])),
        _ => x.attr.get("dtype").or_else(|| x.attr.get("T")).unwrap_or_else(|| panic!("cannot determine dtype for {}", x.op)).clone()
    }
}

fn parse_input(x: &str) -> (&str, usize) {
    match x.find(':') {
        Some(i) => (&x[..i], x[i+1..].parse().unwrap()),
        None => (x, 0)
    }
}

// TODO: use task id?
fn task_name(x: &str) -> String {
    let p = x.rfind('/').expect("unrecognized device name");
    x[..p].to_string()
}
