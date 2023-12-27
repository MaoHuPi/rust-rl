// 2023 (c) MaoHuPi
// rust-rl/src/main.rs
// Referred to [【機器學習2021】概述增強式學習 (Reinforcement Learning, RL)...](https://youtu.be/XWukX-ayIrs?si=LuWwekF-Jq4K2np_)

// new => init => get => set => other
use std::vec::Vec;
use rand::Rng;

mod flexible_network;
use crate::flexible_network::network::*;

fn main() {
    // Main Function
    // Test something important or just fun.
    // Currently, it is an 3-layer random network with cycle connect the hidden layer.
    let mut rng = rand::thread_rng();

    let mut net = Network::new();
    let mut input_layer: Vec<usize> = Vec::new();
    let mut hidden_layer: Vec<usize> = Vec::new();
    let mut output_layer: Vec<usize> = Vec::new();
    for _ in 0..10 {
        input_layer.push(net.new_node(0.0, ActivationFunction::relu));
    }
    for _ in 0..10 {
        let h_id: usize = net.new_node(0.0, ActivationFunction::relu);
        for i_id in input_layer.iter() {
            net.connect(*i_id, h_id, rng.gen_range(-1.0..1.0));
        }
        if h_id%2 == 0{
            net.connect(h_id, h_id, rng.gen_range(-1.0..1.0));
        }
        hidden_layer.push(h_id);
    }
    for _ in 0..3 {
        let o_id: usize = net.new_node(0.0, ActivationFunction::relu);
        for h_id in hidden_layer.iter() {
            net.connect(*h_id, o_id, rng.gen_range(0.0..1.0));
        }
        output_layer.push(o_id);
    }

    net.set_input_id(input_layer);
    net.set_output_id(output_layer);
    net.set_input(Vec::from([5.0; 10]));

    for _ in 0..10 {
        net.next();
        println!("{}", serde_json::to_string(&net.get_output()).unwrap());
    }

    println!("done!");
}