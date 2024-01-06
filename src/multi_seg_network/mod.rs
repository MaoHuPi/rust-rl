// 2024 (c) MaoHuPi
// rust-rl/src/multi_seg_network/mod.rs

use crate::flexible_network::FlexibleNetwork;
use crate::function_segment::FunctionSegment;
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub enum SegmentTypes {
    MultiSegNetwork,
    FlexibleNetwork,
    FunctionSegment,
}
pub trait Segment {
    fn new() -> Self
    where
        Self: Sized;
    fn get_type(self: &Self) -> SegmentTypes;
    fn set_input(self: &mut Self, input: Vec<f64>);
    fn get_output(self: &mut Self) -> Vec<f64>;
    fn export_data(self: &mut Self) -> String;
    fn import_data(self: &mut Self, data: String);
    fn next(self: &mut Self);
    fn can_fitting(self: &Self) -> bool {
        false
    }
    fn fitting(self: &mut Self, anticipated_data: Vec<f64>, rate: f64) {}
    fn can_inverse(self: &Self) -> bool {
        false
    }
    fn inverse(self: &mut Self, data: Vec<f64>, rate: f64) -> Vec<f64> {
        data
    }
}
pub struct MultiSegNetwork {
    segments: Vec<Box<(dyn Segment + 'static)>>,
    input_value: Vec<f64>,
    output_value: Vec<f64>,
}
#[derive(Serialize, Deserialize)]
struct MultiSegNetworkData {
    types: Vec<SegmentTypes>,
    data: Vec<String>,
}
#[allow(dead_code)]
impl Segment for MultiSegNetwork {
    fn new() -> Self {
        Self {
            segments: Vec::new(),
            input_value: Vec::new(),
            output_value: Vec::new(),
        }
    }
    fn get_type(self: &Self) -> SegmentTypes {
        SegmentTypes::MultiSegNetwork
    }
    fn set_input(self: &mut Self, input: Vec<f64>) {
        self.input_value = input;
    }
    fn get_output(self: &mut Self) -> Vec<f64> {
        self.output_value.clone()
    }
    fn export_data(self: &mut Self) -> String {
        let mut data = MultiSegNetworkData {
            types: Vec::new(),
            data: Vec::new(),
        };
        for i in 0..self.segments.len() {
            data.types.push(self.segments[i].get_type());
            data.data.push(self.segments[i].export_data());
        }
        serde_json::to_string(&data).unwrap()
    }
    fn import_data(self: &mut Self, data: String) {
        let data: MultiSegNetworkData = serde_json::from_str(&data.as_str()).unwrap();
        let mut segments: Vec<Box<(dyn Segment + 'static)>> = Vec::new();
        for i in 0..data.types.len() {
            let mut seg = MultiSegNetwork::new_seg(data.types[i].clone());
            (*seg).import_data(data.data[i].clone());
            segments.push(seg);
        }
        self.segments = segments;
    }
    fn next(self: &mut Self) {
        let mut value: Vec<f64> = self.input_value.clone();
        for i in 0..self.segments.len() {
            self.segments[i].set_input(value);
            self.segments[i].next();
            value = self.segments[i].get_output();
        }
        self.output_value = value;
    }
}
#[allow(dead_code)]
impl MultiSegNetwork {
    pub fn new_seg(seg_type_name: SegmentTypes) -> Box<dyn Segment + 'static> {
        match seg_type_name {
            SegmentTypes::MultiSegNetwork => Box::new(MultiSegNetwork::new()),
            SegmentTypes::FlexibleNetwork => Box::new(FlexibleNetwork::new()),
            SegmentTypes::FunctionSegment => Box::new(FunctionSegment::new()),
        }
    }
    pub fn push_seg(self: &mut Self, segment: (impl Segment + 'static)) -> usize {
        self.segments.push(Box::new(segment));
        self.segments.len() - 1
    }
    pub fn operate_seg(
        self: &mut Self,
        id: usize,
        call_back: impl FnOnce(&mut Box<(dyn Segment + 'static)>),
    ) {
        call_back(&mut self.segments[id]);
    }
}
pub mod flexible_network;
pub mod function_segment;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn test_multi_seg_network_set_input_give_a_vector_it_should_be_set_to_input_value_member() {
        const INPUT_VALUE: [f64; 3] = [1.0, 2.0, 3.0];
        let mut multi_seg: MultiSegNetwork = MultiSegNetwork::new();

        multi_seg.set_input(Vec::from(INPUT_VALUE));

        assert_eq!(multi_seg.input_value, Vec::from(INPUT_VALUE));
    }
}
