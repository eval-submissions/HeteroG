use crate::graph::*;

pub trait Strategy {
    fn resolve();
}

pub struct Naive;

impl Strategy for Naive {
    fn resolve() {
        unimplemented!()
    }
}
