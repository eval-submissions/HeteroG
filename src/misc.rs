use crate::graph::Form;
use crate::proto::{graph::GraphDef, node_def::NodeDef, attr_value::AttrValue, types::DataType};
use std::collections::BTreeMap;

pub struct Target {
    pub pb: GraphDef,
    pub devices: Box<[String]>,
    pub links: Box<[u64]>, // the bandwidth of each link
    pub paths: Box<[Box<[usize]>]>, // the i*n+j element is the links that i->j uses (currently only one path between each pair)
    pub sinks: Box<[String]>, // sink nodes
    pub nccls: BTreeMap<String, [f64; 4]> // the key is a comma separated sorted list of device names, the values are [coef1, interc1, coef2, interc2]. The model is time = max( coef1 * size + interc1, coef2 * size + interc2 ). The size unit is KB.
}

impl Target {
    pub fn new(pb: GraphDef, devices: Box<[String]>, links: Box<[u64]>, paths: Box<[Box<[usize]>]>, sinks: Box<[String]>, nccls: BTreeMap<String, [f64; 4]>) -> Self {
        Target { pb, devices, links, paths, sinks, nccls }
    }

    pub fn ndev(&self) -> usize {
        self.devices.len()
    }
}

pub trait Profiler {
    fn profile(&self, node: &NodeDef, device_id: usize) -> Option<u64>;
}

pub struct DataProfiler {
    /// the value is a binary sorted array contains replica_number and the time required on each device given replicated by that number
    pub data: BTreeMap<String, Vec<(usize, Vec<u64>)>>
}

impl Profiler for DataProfiler {
    fn profile(&self, node: &NodeDef, device_id: usize) -> Option<u64> {
        let origin_name = node.attr.get("_tge_origin")?.get_s();
        // technically we do not need to extract the form if we use a profiler since it will be reflected by the input size.
        let form = Form::from_code(std::str::from_utf8(node.attr.get("_tge_form")?.get_s()).ok()?);
        let nrep = if form.is_part() {
            form.ndev()
        } else {
            1
        };

        let prof = self.data.get(&String::from_utf8(origin_name.to_vec()).unwrap())?;
        let time = match prof.binary_search_by_key(&nrep, |x| x.0) {
            Ok(i) => prof[i].1[device_id],
            Err(i) => if i >= prof.len() {
                prof[i - 1].1[device_id]
            } else {
                prof[i].1[device_id]
            }
        };

        Some(time)
    }
}
