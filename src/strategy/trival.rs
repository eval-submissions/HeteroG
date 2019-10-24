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
        for node in graph.nodes.iter_mut() {
            node.put_on_device(0);
        }
    }
}
