use oh_my_rust::*;
use crate::graph::*;

// if we do not remove these, we need to modify this field so that it has the correct node name of replicated operators
pub fn remove_collocation_hint(target: &mut Target) {
    for node in target.pb.node.iter_mut() {
        if let Some(x) = node.attr.get_mut("_class") {
            if let Some(crate::proto::attr_value::AttrValue_oneof_value::list(ref mut list)) = &mut x.value {
                list.s = list.s.iter().filter(|x| !x.starts_with(b"loc:")).cloned().collect()
            }
        }
    }
}

pub fn remove_shape_hint(target: &mut Target) {
    for node in target.pb.node.iter_mut() {
        node.attr.remove("_output_shapes");
    }
}

pub fn remove_dangling_nodes(target: &mut Target) {
    let sinks: Vec<_> = target.sinks.iter().map(|x| format!("{}/replica_0", x)).collect();

    // note: don't forget control dependency
    let dict: std::collections::HashMap<_, Vec<_>> = target.pb.node.iter().map(|node| {
        (&node.name[..], node.input.iter().map(|x| {
            if x.starts_with('^') {
                return &x[1..]
            }
            match x.find(':') {
                Some(i) => &x[..i],
                None => x
            }
        }).collect())
    }).collect();
    let mut keep = std::collections::HashSet::new();
    let mut queue: std::collections::VecDeque<_> = sinks.iter().map(|x| &x[..]).collect();

    while let Some(x) = queue.pop_front() {
        if keep.insert(x.to_string()) {
            queue.extend(&dict[x]);
        }
    }

    // hacky way to avoid clone
    let mut x = std::mem::replace(&mut target.pb.node, vec![].into()).into_vec();
    x.retain(|x| keep.contains(&x.name[..]));
    std::mem::replace(&mut target.pb.node, x.into());
}

pub fn destruct_names(target: &mut Target) {
    for node in target.pb.node.iter_mut() {
        node.name = node.name.replace('/', "__");
        for input in node.input.iter_mut() {
            *input = input.replace('/', "__");
        }
    }
}
