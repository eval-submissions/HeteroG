use crate::strategy::Strategy;
use crate::graph::Target;

pub struct NotAtAll;

type Graph = crate::graph::Graph<(), ()>;
type Node = crate::graph::Node<(), ()>;
type Tensor = crate::graph::Tensor<(), ()>;

impl Strategy for NotAtAll {
    type NEX = ();
    type TEX = ();

    #[allow(clippy::cast_ref_to_mut)]
    fn plan(&mut self, graph: &mut Graph, _target: &mut Target) {
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
