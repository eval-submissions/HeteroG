use std::convert::TryInto;

use crate::graph::*;
use crate::proto::types::DataType;
use crate::proto::attr_value::{AttrValue, AttrValue_oneof_value};
use crate::proto::node_def::NodeDef;
use crate::proto::tensor::TensorProto;

pub trait Strategy {
    type NEX: Default;
    type TEX: Default;
    /// make the plan, setting the neccesary fields for nodes and tensors and create the aux nodes on target
    fn plan(&mut self, graph: &mut Graph<Self::NEX, Self::TEX>, target: &mut Target);
}

mod trival;
pub use trival::NotAtAll;

// mod dp; // data parallel
// pub use dp::{DataParallelOneForAll, DataParallelNccl, DataParallelRing};

// mod heft; // earliest finish time
// pub use heft::NaiveGreedyEarliestFinishTime;

// mod dp2; // dynamic programming
// pub use dp2::DynamicProgrammingEarliestFinishTime;

mod custom;
pub use custom::Custom;

// mod grouped_custom;
// pub use grouped_custom::GroupedCustom;
