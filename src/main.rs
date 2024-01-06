/*
 * 2024 (c) MaoHuPi
 * rust-rl/src/main.rs
 * Referred to [【機器學習2021】概述增強式學習 (Reinforcement Learning, RL)...](https://youtu.be/XWukX-ayIrs?si=LuWwekF-Jq4K2np_)
 */

use std::iter;
use std::vec::Vec;
use rand::Rng;
use std::fs::File;
use std::io::prelude::*;
use std::time::SystemTime;

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
    let mut net_data: NetworkData;

    /* train and save modle */
    fn try_fitting() {
        // Main Function
        // Test something important or just fun.
        // Currently, it is an 3-layer random network with cycle connect the hidden layer.
        let mut rng: rand::prelude::ThreadRng = rand::thread_rng();
        let test_data: [[f64; 6]; 3] = [[1.0, 2.0, 3.0, -1.0, 2.0, 0.5], [2.0, 4.0, 6.0, -2.0, 8.0, 0.5], [1.5, 1.2, 2.7, 0.3, 1.8, 1.25]];
        let mut net = Network::new();
        let input_layer: usize = net.new_layer(2, 0.0, ActivationFunctionEnum::DoNothing);
        let hidden_layer_list: Vec<usize> = vec![false; 20].iter().map(|n| net.new_layer(10, 0.0, ActivationFunctionEnum::DoNothing)).collect::<Vec<usize>>();
        let output_layer: usize = net.new_layer(1, rng.gen_range(-1.0..1.0), ActivationFunctionEnum::DoNothing);
        // net.connect_layer(input_layer, output_layer, 0.1);
        net.connect_layer(input_layer, hidden_layer_list[0], 0.1);
        for i in 0..hidden_layer_list.len()-1 {
            net.connect_layer(hidden_layer_list[i], hidden_layer_list[i+1], 0.1);
        }
        net.connect_layer(hidden_layer_list[hidden_layer_list.len()-1], output_layer, 0.1);
        net.set_input_layer(input_layer);
        net.set_output_layer(output_layer);
        
        for rate in [0.000006] {
            for _ in 1..=1000 {
                let input_data: Vec <f64> = Vec::from([rng.gen_range(0..10) as f64, rng.gen_range(0..10) as f64]);
                let anticipated_data: Vec<f64> = Vec::from([input_data[0] + input_data[1]]);
                net.set_input(input_data);
                net.next();
                if (net.get_output()[0]).is_nan() {
                    panic!("NaN! Check if learning rate is to large.");
                }
                net.fitting(anticipated_data, rate);
            }
        }
        
        for test_pair in test_data {
            net.set_input(Vec::from(&test_pair[0..=1]));
            net.next();
            let output = net.get_output();
            println!("{}+{} = {} (must be {}, delta {})", test_pair[0], test_pair[1], output[0], test_pair[2], (test_pair[2]-output[0]).abs());
        }

        let mut file = File::create("model/net.json").unwrap();
        let _  = file.write_all(serde_json::to_string(&net.export_data()).unwrap().as_bytes());
        
        println!("done!");
    }
    // until_ok!(try_fitting);
    let start_time: SystemTime = SystemTime::now();
    try_fitting();
    match start_time.elapsed() {
        Ok(elapsed) => { println!("use time: {}ms", elapsed.as_millis()); }
        Err(_) => {}
    }
    

    /* load modle */
    // let mut file = File::open("model/net.json").unwrap();
    // let mut content = String::new();
    // file.read_to_string(&mut content).unwrap();
    // let net_data: NetworkData = serde_json::from_str(content.as_str()).unwrap();

    // let mut net = Network::new();
    // net.import_data(net_data);
    // net.set_input(Vec::from([1.0, 1.0]));
    // net.next();
    // println!("output: {}", net.get_output()[0]);


    /* generate maze map */
    // let maze: Vec<Vec<usize>> = generate_maze(10, 10);
    // let map = maze.iter().map(|line| {
    //     line.iter().map(|&value| {
    //         value.to_string()
    //     }).collect::<Vec<String>>().join("")
    // }).collect::<Vec<String>>().join("\n");
    // // let map: String = serde_json::to_string(&maze).unwrap();
    // print!("{}", map);
}