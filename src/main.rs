// 2023 (c) MaoHuPi
// rust-rl/src/main.rs
// Referred to [【機器學習2021】概述增強式學習 (Reinforcement Learning, RL)...](https://youtu.be/XWukX-ayIrs?si=LuWwekF-Jq4K2np_)

use std::ptr::null;
// new => init => get => set => other
use std::vec::Vec;
use rand::Rng;

mod flexible_network;
use crate::flexible_network::network::*;

mod maze_game;
use crate::maze_game::*;

fn main() {
    // Main Function
    // Test something important or just fun.
    // Currently, it is an 3-layer random network with cycle connect the hidden layer.
    let mut rng = rand::thread_rng();

    let mut net = Network::new();
    let mut input_layer: Vec<usize> = Vec::new();
    let mut hidden_layer: Vec<usize> = Vec::new();
    let mut output_layer: Vec<usize> = Vec::new();
    for _ in 0..2 {
        input_layer.push(net.new_node(0.0, ActivationFunctionEnum::Sigmoid));
    }
    for _ in 0..10 {
        let h_id: usize = net.new_node(0.0, ActivationFunctionEnum::Sigmoid);
        for i_id in input_layer.iter() {
            net.connect(*i_id, h_id, rng.gen_range(-1.0..1.0));
        }
        if h_id%2 == 0{
            net.connect(h_id, h_id, rng.gen_range(-1.0..1.0));
        }
        hidden_layer.push(h_id);
    }
    for _ in 0..1 {
        let o_id: usize = net.new_node(0.0, ActivationFunctionEnum::Sigmoid);
        for h_id in hidden_layer.iter() {
            net.connect(*h_id, o_id, rng.gen_range(0.0..1.0));
        }
        output_layer.push(o_id);
    }
    net.set_input_id(input_layer);
    net.set_output_id(output_layer);

    for _ in 0..1 {
        // for i in 0..10 {
            // let i_f64: f64 = f64::try_from(i).unwrap();
            let input_value: Vec<f64> = Vec::from([rng.gen_range(0.0..=10.0)/10.0, rng.gen_range(0.0..=10.0)/10.0]);
            let output_value: Vec<f64> = Vec::from([input_value[0] + input_value[1]]);
            println!("{} {} {}", input_value[0],  input_value[1], output_value[0]);
            net.set_input(Vec::from(input_value));
            net.next();
            net.fitting(output_value, 0.0001);
        // }
    }
    
    net.set_input(Vec::from([1.0, 1.0]));
    net.next();
    let output_value: Vec<f64> = net.get_output();
    println!("{}, {}", serde_json::to_string(&output_value).unwrap(), output_value[0].is_nan());

    println!("done!");
}