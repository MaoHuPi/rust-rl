// 2023 (c) MaoHuPi
// rust-rl/src/main.rs
// Referred to [【機器學習2021】概述增強式學習 (Reinforcement Learning, RL)...](https://youtu.be/XWukX-ayIrs?si=LuWwekF-Jq4K2np_)

use std::vec::Vec;
use rand::Rng;

mod flexible_network;
use crate::flexible_network::network::*;

mod maze_game;
use crate::maze_game::*;

macro_rules! until_ok {
    ($do_something: expr) => {
        loop {
            let result = std::panic::catch_unwind(|| $do_something());
            if result.is_ok() {
                break;
            }
        }
    };
}

fn main() {
    fn try_fitting() {
        // Main Function
        // Test something important or just fun.
        // Currently, it is an 3-layer random network with cycle connect the hidden layer.
        let mut rng = rand::thread_rng();

        let mut net = Network::new();
        let mut input_layer: Vec<usize> = Vec::new();
        let mut hidden_layer: Vec<usize> = Vec::new();
        let mut output_layer: Vec<usize> = Vec::new();
        for _ in 0..2 {
            input_layer.push(net.new_node(0.0, ActivationFunctionEnum::DoNothing));
        }
        for _ in 0..5 {
            let h_id: usize = net.new_node(0.0, ActivationFunctionEnum::ReLU);
            for i_id in input_layer.iter() {
                // net.connect(*i_id, h_id, rng.gen_range(-1.0..1.0));
                net.connect(*i_id, h_id, 1.0);
            }
            // if h_id%2 == 0{
            //     net.connect(h_id, h_id, rng.gen_range(-1.0..1.0));
            // }
            hidden_layer.push(h_id);
        }
        for _ in 0..1 {
            let o_id: usize = net.new_node(0.0, ActivationFunctionEnum::ReLU);
            for h_id in hidden_layer.iter() {
                // net.connect(*h_id, o_id, rng.gen_range(0.0..1.0));
                net.connect(*h_id, o_id, 1.0);
            }
            output_layer.push(o_id);
        }
        net.set_input_id(input_layer);
        net.set_output_id(output_layer.clone());
        
        for _ in 0..100000 {
            let input_value: Vec<f64> = Vec::from([rng.gen_range(0.0..=1.0), rng.gen_range(0.0..=1.0)]);
            let output_value: Vec<f64> = Vec::from([input_value[0] * input_value[1]]);
            net.set_input(Vec::from(input_value));
            net.next();
            net.fitting(Vec::from(output_value), 0.001);
        }
        
        net.set_input(Vec::from([1.0, 0.5]));
        net.next();
        let output_value: Vec<f64> = net.get_output();
        println!("{}, {}", serde_json::to_string(&output_value).unwrap(), output_value[0].is_nan());
        net.set_input(Vec::from([2.0, 1.5]));
        net.next();
        let output_value: Vec<f64> = net.get_output();
        println!("{}, {}", serde_json::to_string(&output_value).unwrap(), output_value[0].is_nan());
        net.set_input(Vec::from([0.1, 0.5]));
        net.next();
        let output_value: Vec<f64> = net.get_output();
        println!("{}, {}", serde_json::to_string(&output_value).unwrap(), output_value[0].is_nan());
        let trained_network: NetworkData = net.get_data();
        println!("{}", serde_json::to_string(&trained_network).unwrap());

        println!("done!");
    };
    // until_ok!(try_fitting);
    try_fitting();
}