use crate::graph::*;

// if we do not remove these, we need to modify this field so that it has the correct node name of replicated operators
pub fn remove_colocation_hint(target: &mut Target) {
    for node in target.pb.node.iter_mut() {
        if let Some(x) = node.attr.get_mut("_class".into()) {
            if let Some(crate::proto::attr_value::AttrValue_oneof_value::list(ref mut list)) = &mut x.value {
                list.s = list.s.iter().filter(|x| !x.starts_with(b"loc:")).map(|x| x.clone()).collect()
            }
        }
    }
}

pub fn remove_dangling_nodes(_target: &mut Target) {
    unimplemented!()
}

pub fn fix_special_controllers(_target: &mut Target) {
    // fix the GradientDescent and init controler
    unimplemented!()
}

pub fn destructify_names(target: &mut Target) {
    for node in target.pb.node.iter_mut() {
        node.name = node.name.replace('/', "__");
        for input in node.input.iter_mut() {
            *input = input.replace('/', "__");
        }
    }
}
