/*
 * 2023 (c) MaoHuPi
 * rust-rl/src/main.rs
 * Referred to [【機器學習2021】概述增強式學習 (Reinforcement Learning, RL)...](https://youtu.be/XWukX-ayIrs?si=LuWwekF-Jq4K2np_)
 */

use std::vec::Vec;
use rand::Rng;
use std::fs::File;
use std::io::prelude::*;

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
    // fn try_fitting() {
    //     // Main Function
    //     // Test something important or just fun.
    //     // Currently, it is an 3-layer random network with cycle connect the hidden layer.
    //     let mut rng: rand::prelude::ThreadRng = rand::thread_rng();
    //     let test_data: [[f64; 3]; 3] = [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.5, 1.2, 2.7]];
    //     let mut net = Network::new();
    //     let mut input_layer: usize = net.new_layer(2, 0.0, ActivationFunctionEnum::DoNothing);
    //     let mut hidden_layer: usize = net.new_layer(5, 0.0, ActivationFunctionEnum::DoNothing);
    //     let mut output_layer: usize = net.new_layer(1, 0.0, ActivationFunctionEnum::ReLU);
    //     net.connect_layer(input_layer, hidden_layer, 1.0);
    //     net.connect_layer(hidden_layer, output_layer, 1.0);
    //     net.set_input_layer(input_layer);
    //     net.set_output_layer(output_layer);
        
    //     for rate in [0.001, 0.0001] {
    //         for _ in 1..=10000 {
    //             // for test_pair in test_data {
    //             //     net.set_input(Vec::from(&test_pair[0..=1]));
    //             //     net.next();
    //             //     net.fitting(Vec::from([test_pair[2]]), rate);
    //             // }

    //             let input_data: Vec <f64> = Vec::from([rng.gen_range(0.0..10.0), rng.gen_range(0.0..10.0)]);
    //             let anticipated_data: Vec<f64> = Vec::from([input_data[0] + input_data[1]]);
    //             net.set_input(input_data);
    //             net.next();
    //             net.fitting(anticipated_data, rate);
    //         }
    //     }
        
    //     for test_pair in test_data {
    //         net.set_input(Vec::from(&test_pair[0..=1]));
    //         net.next();
    //         println!("{} {}", net.get_output()[0], test_pair[2]);
    //     }

    //     let mut file = File::create("model/net.json").unwrap();
    //     let _  = file.write_all(serde_json::to_string(&net.export_data()).unwrap().as_bytes());
        
    //     println!("done!");
    // };
    // until_ok!(try_fitting);
    // try_fitting();
    

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
    let maze: Vec<Vec<usize>> = generate_maze(10, 10);
    let map = maze.iter().map(|line| {
        line.iter().map(|&value| {
            value.to_string()
        }).collect::<Vec<String>>().join("")
    }).collect::<Vec<String>>().join("\n");
    // let map: String = serde_json::to_string(&maze).unwrap();
    print!("{}", map);
}