/*
 * 2024 (c) MaoHuPi
 * rust-rl/src/multi_seg_network/function_segment.rs
 */

use serde::{Deserialize, Serialize};
// Make the customize struct be able to json stringify

use crate::multi_seg_network::{Segment, SegmentTypes};

#[derive(Clone, Copy, Serialize, Deserialize)]
pub enum FunctionSegmentFunctionEnum {
    DoNothing,
    Fraction,
    SoftMax,
}
pub struct FunctionSegment {
    function: fn(Vec<f64>) -> Vec<f64>,
    function_enum: FunctionSegmentFunctionEnum,
    input_value: Vec<f64>,
    output_value: Vec<f64>,
}
#[derive(Serialize, Deserialize)]
struct FunctionSegmentData {
    fs_fn: FunctionSegmentFunctionEnum,
}
fn do_nothing(input: Vec<f64>) -> Vec<f64> {
    input
}
fn fraction(input: Vec<f64>) -> Vec<f64> {
    let sum: f64 = input.iter().sum();
    input.iter().map(|n| n / sum).collect::<Vec<f64>>()
}
fn soft_max(input: Vec<f64>) -> Vec<f64> {
    let natural_exponential_input: Vec<f64> = input.iter().map(|&n| std::f64::consts::E.powf(n)).collect::<Vec<f64>>();
    fraction(natural_exponential_input)
}
#[allow(dead_code)]
impl Segment for FunctionSegment {
    fn new() -> Self {
        Self {
            function: do_nothing,
            function_enum: FunctionSegmentFunctionEnum::DoNothing,
            input_value: Vec::new(),
            output_value: Vec::new(),
        }
    }
    fn get_type(self: &Self) -> SegmentTypes {
        SegmentTypes::FunctionSegment
    }
    fn set_input(self: &mut Self, input: Vec<f64>) {
        self.input_value = input;
    }
    fn get_output(self: &mut Self) -> Vec<f64> {
        self.output_value.clone()
    }
    fn export_data(self: &mut Self) -> String {
        let mut data = FunctionSegmentData {
            fs_fn: self.function_enum,
        };
        serde_json::to_string(&data).unwrap()
    }
    fn import_data(self: &mut Self, data: String) {
        let data: FunctionSegmentData = serde_json::from_str(&data.as_str()).unwrap();
        self.set_function(data.fs_fn);
    }
    fn next(self: &mut Self) {
        self.output_value = (self.function)(self.input_value.clone());
    }
}
impl FunctionSegment {
    fn get_function(function_enum: FunctionSegmentFunctionEnum) -> fn(Vec<f64>) -> Vec<f64> {
        match function_enum {
            FunctionSegmentFunctionEnum::DoNothing => do_nothing,
            FunctionSegmentFunctionEnum::Fraction => fraction,
            FunctionSegmentFunctionEnum::SoftMax => soft_max,
        }
    }
    pub fn set_function(self: &mut Self, function_enum: FunctionSegmentFunctionEnum) {
        self.function_enum = function_enum;
        self.function = Self::get_function(self.function_enum);
    }
}