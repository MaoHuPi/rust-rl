/*
 * 2024 (c) MaoHuPi
 * rust-rl/src/main.rs
 * Referred to [【機器學習2021】概述增強式學習 (Reinforcement Learning, RL)...](https://youtu.be/XWukX-ayIrs?si=LuWwekF-Jq4K2np_)
 */

use rand::Rng;
use std::fs::File;
use std::io::prelude::*;
use std::iter;
use std::time::SystemTime;
use std::vec::Vec;
use colored::Colorize;

mod multi_seg_network;
use crate::multi_seg_network::*;
use crate::multi_seg_network::flexible_network::{FlexibleNetwork, ActivationFunctionEnum};
use crate::multi_seg_network::function_segment::{FunctionSegment, FunctionSegmentFunctionEnum};

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
    let start_time: SystemTime = SystemTime::now();
    let mut net_data: String;

    /* train and save modle */
    // fn try_fitting() {
    //     // Main Function
    //     // Test something important or just fun.
    //     // Currently, it is an 3-layer random network with cycle connect the hidden layer.
    //     let mut rng: rand::prelude::ThreadRng = rand::thread_rng();
    //     let test_data: [[f64; 6]; 3] = [
    //         [1.0, 2.0, 3.0, -1.0, 2.0, 0.5],
    //         [2.0, 4.0, 6.0, -2.0, 8.0, 0.5],
    //         [1.5, 1.2, 2.7, 0.3, 1.8, 1.25],
    //     ];
    //     let mut flexible_net = FlexibleNetwork::new();
    //     let input_layer: usize = flexible_net.new_layer(2, 0.0, ActivationFunctionEnum::DoNothing);
    //     let hidden_layer_list: Vec<usize> = vec![false; 2]
    //         .iter()
    //         .map(|n| flexible_net.new_layer(2, 0.0, ActivationFunctionEnum::DoNothing))
    //         .collect::<Vec<usize>>();
    //     let output_layer: usize = flexible_net.new_layer(1, 0.0, ActivationFunctionEnum::DoNothing);
    //     // net.connect_layer(input_layer, output_layer, 0.1);
    //     flexible_net.connect_layer(input_layer, hidden_layer_list[0], 0.1);
    //     for i in 0..hidden_layer_list.len() - 1 {
    //         flexible_net.connect_layer(hidden_layer_list[i], hidden_layer_list[i + 1], 0.1);
    //     }
    //     flexible_net.connect_layer(
    //         hidden_layer_list[hidden_layer_list.len() - 1],
    //         output_layer,
    //         0.1,
    //     );
    //     flexible_net.set_input_layer(input_layer);
    //     flexible_net.set_output_layer(output_layer);

    //     let mut output_function: FunctionSegment = FunctionSegment::new();
    //     output_function.set_function(FunctionSegmentFunctionEnum::SoftMax);

    //     let mut multi_seg: MultiSegNetwork = MultiSegNetwork::new();
    //     let flexible_net_id = multi_seg.push_seg(flexible_net);
    //     multi_seg.push_seg(output_function);

    //     for rate in [0.00062] {
    //         for _ in 1..=100 {
    //             let input_data: Vec<f64> =
    //                 Vec::from([rng.gen_range(0..10) as f64, rng.gen_range(0..10) as f64]);
    //             let anticipated_data: Vec<f64> = Vec::from([input_data[0] + input_data[1]]);
    //             multi_seg.set_input(input_data);
    //             multi_seg.next();
    //             if (multi_seg.get_output()[0]).is_nan() {
    //                 panic!("NaN! Check if learning rate is to large.");
    //             }
    //             multi_seg.operate_seg(flexible_net_id, |seg| {
    //                 seg.fitting(anticipated_data, rate);
    //             });
    //         }
    //     }

    //     for test_pair in test_data {
    //         multi_seg.set_input(Vec::from(&test_pair[0..=1]));
    //         multi_seg.next();
    //         multi_seg.operate_seg(flexible_net_id, |seg| {
    //             let output = seg.get_output();
    //             println!(
    //                 "{}: {}+{} = {} (must be {}, delta {})",
    //                 "before SoftMax".yellow(),
    //                 test_pair[0],
    //                 test_pair[1],
    //                 output[0],
    //                 test_pair[2],
    //                 (test_pair[2] - output[0]).abs()
    //             );
    //         });
    //         let output = multi_seg.get_output();
    //         println!(
    //             "{}: {}+{} = {} (must be {}, delta {})",
    //             "after SoftMax".green(), 
    //             test_pair[0],
    //             test_pair[1],
    //             output[0],
    //             test_pair[2],
    //             (test_pair[2] - output[0]).abs()
    //         );
    //     }

    //     let mut file = File::create("model/net.json").unwrap();
    //     let _ = file.write_all(multi_seg.export_data().as_bytes());

    //     println!("done!");
    // }
    // // until_ok!(try_fitting);
    // try_fitting();

    /* load modle */
    let mut file: File = File::open("model/net.json").unwrap();
    let mut content = String::new();
    file.read_to_string(&mut content).unwrap();
    let net_data: String = content.to_string();

    let mut net: MultiSegNetwork = MultiSegNetwork::new();
    net.import_data(net_data);
    net.set_input(Vec::from([1.0, 1.5]));
    net.next();
    println!("output: {}", net.get_output()[0]);

    /* generate maze map */
    // let maze: Vec<Vec<usize>> = generate_maze(10, 10);
    // let map = maze.iter().map(|line| {
    //     line.iter().map(|&value| {
    //         value.to_string()
    //     }).collect::<Vec<String>>().join("")
    // }).collect::<Vec<String>>().join("\n");
    // // let map: String = serde_json::to_string(&maze).unwrap();
    // print!("{}", map);

    match start_time.elapsed() {
        Ok(elapsed) => {
            println!("use time: {}ms", elapsed.as_millis());
        }
        Err(_) => {}
    }
}
