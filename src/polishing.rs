use oh_my_rust::*;
use crate::graph::*;

// if we do not remove these, we need to modify this field so that it has the correct node name of replicated operators
pub fn remove_colocation_hint(target: &mut Target) {
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

pub fn remove_dangling_nodes(end_points: &[&str], target: &mut Target) {
    // note: don't forget control dependency
    let dict: std::collections::BTreeMap<_, Vec<_>> = target.pb.node.iter().map(|node| {
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
    let mut keep = std::collections::BTreeSet::new();
    let mut queue: std::collections::VecDeque<_> = end_points.iter().copied().collect();

    while let Some(x) = queue.pop_front() {
        if keep.insert(x) {
            queue.extend(&dict[x]);
        }
    }

    target.pb.node = target.pb.node.clone().into_iter().filter(|x| keep.contains(&x.name[..])).collect() // TODO: no clone?
}

pub fn destructify_names(target: &mut Target) {
    for node in target.pb.node.iter_mut() {
        node.name = node.name.replace('/', "__");
        for input in node.input.iter_mut() {
            *input = input.replace('/', "__");
        }
    }
}
