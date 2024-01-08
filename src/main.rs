/*
 * 2024 (c) MaoHuPi
 * rust-rl/src/main.rs
 * Referred to [【機器學習2021】概述增強式學習 (Reinforcement Learning, RL)...](https://youtu.be/XWukX-ayIrs?si=LuWwekF-Jq4K2np_)
 */

use colored::Colorize;
use getch::Getch;
use rand::Rng;
use std::collections::btree_set;
use std::fs::File;
use std::io;
use std::io::prelude::*;
use std::iter;
use std::path::Path;
use std::time::SystemTime;
use std::vec::Vec;

mod multi_seg_network;
use crate::multi_seg_network::flexible_network::{ActivationFunctionEnum, FlexibleNetwork};
use crate::multi_seg_network::function_segment::{FunctionSegment, FunctionSegmentFunctionEnum};
use crate::multi_seg_network::*;

mod maze_game;
use crate::maze_game::*;

const MODEL_PATH: &str = "model/net.json";

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
fn random_as_probability<T>(value_list: Vec<T>, probability_list: Vec<f64>) -> T
where
    T: Clone,
{
    let mut sum: f64 = 0.0;
    let mut case_segment_list: Vec<(f64, T)> = Vec::new();
    for i in 0..probability_list.len() {
        sum += probability_list[i];
        case_segment_list.push((sum, value_list[i].clone()));
    }
    let mut rng: rand::prelude::ThreadRng = rand::thread_rng();
    let random_result: f64 = rng.gen_range(0.0..sum);
    let mut i: usize = 0;
    loop {
        if i > case_segment_list.len() - 1 {
            break case_segment_list[0].1.clone();
        }
        if random_result < case_segment_list[i].0 {
            break case_segment_list[i].1.clone();
        }
        i += 1;
    }
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
    // let mut file: File = File::open("model/net.json").unwrap();
    // let mut content = String::new();
    // file.read_to_string(&mut content).unwrap();
    // let net_data: String = content.to_string();

    // let mut net: MultiSegNetwork = MultiSegNetwork::new();
    // net.import_data(net_data);
    // net.set_input(Vec::from([1.0, 1.5]));
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

    /* maze game */
    // let mut game: Game = Game::new();
    // game.set_screen_size([21, 21]);
    // game.set_maze_size([10, 10]);
    // game.start();
    // while game.playing() {
    //     let screen: String = game.get_screen_string();
    //     println!("");
    //     println!("{}", screen);
    //     let mut action_key: u8 = 0;
    //     let g = Getch::new();
    //     action_key = g.getch().unwrap();
    //     game.action(match action_key {
    //         3 => break,
    //         // Press [Ctrl] + [C] to exit the game.
    //         27 => break,
    //         // Press [Escape] to exit the game.
    //         119 => GameAction::Up,
    //         // Press [W] to go up.
    //         115 => GameAction::Down,
    //         // Press [S] to go down.
    //         97 => GameAction::Left,
    //         // Press [A] to go left.
    //         100 => GameAction::Right,
    //         // Press [D] to go right.
    //         _ => GameAction::Hold,
    //         // Press other key just do nothing.
    //     });
    //     io::stdout().flush().unwrap();
    // }
    // println!("score: {}", game.get_score());

    // match start_time.elapsed() {
    //     Ok(elapsed) => {
    //         println!("use time: {}ms", elapsed.as_millis());
    //     }
    //     Err(_) => {}
    // }

    fn try_fitting() {
        let mut multi_seg: MultiSegNetwork = MultiSegNetwork::new();
        let mut flexible_net_id: usize = 0;
        if Path::new(MODEL_PATH).exists() {
            let mut file: File = File::open(MODEL_PATH).unwrap();
            let mut content = String::new();
            file.read_to_string(&mut content).unwrap();
            let net_data: String = content.to_string();
            multi_seg.import_data(net_data);
            flexible_net_id = 0;
        } else {
            let mut flexible_net = FlexibleNetwork::new();
            let input_layer: usize =
                flexible_net.new_layer(5, 0.0, ActivationFunctionEnum::DoNothing);
            let hidden_layer_list: Vec<usize> = vec![false; 2]
                .iter()
                .map(|n| flexible_net.new_layer(5, 0.0, ActivationFunctionEnum::ReLU))
                .collect::<Vec<usize>>();
            let output_layer: usize =
                flexible_net.new_layer(5, 0.0, ActivationFunctionEnum::ReLU);
            flexible_net.connect_layer(input_layer, hidden_layer_list[0], 0.1);
            for i in 0..hidden_layer_list.len() - 1 {
                flexible_net.connect_layer(hidden_layer_list[i], hidden_layer_list[i + 1], 0.1);
            }
            flexible_net.connect_layer(
                hidden_layer_list[hidden_layer_list.len() - 1],
                output_layer,
                0.1,
            );
            flexible_net.set_input_layer(input_layer);
            flexible_net.set_output_layer(output_layer);

            let mut output_function: FunctionSegment = FunctionSegment::new();
            output_function.set_function(FunctionSegmentFunctionEnum::SoftMax);

            flexible_net_id = multi_seg.push_seg(flexible_net);
            multi_seg.push_seg(output_function);
        }

        // let rate: f64 = 0.00062;
        let rate: f64 = 0.00001;
        for _ in 0..1000 {
            let mut data_pair_list: Vec<[Vec<f64>; 2]> = Vec::new();

            let mut game: Game = Game::new();
            game.set_screen_size([5, 5]);
            game.set_maze_size([2, 2]);
            game.start();
            while game.playing() {
                if game.get_step_count() > 10 {
                    break;
                }
                // let screen_string: String = game.get_screen_string();
                // println!("");
                // println!("{}", screen_string);
                let screen: Vec<Vec<ScreenElement>> = game.get_screen();
                let input_data = screen
                    .iter()
                    .map(|row| {
                        row.iter()
                            .map(|e| match e {
                                ScreenElement::Empty => 0.0,
                                ScreenElement::Road => 1.0,
                                ScreenElement::Wall => 2.0,
                                ScreenElement::StartPoint => 3.0,
                                ScreenElement::DestinationPoint => 4.0,
                                ScreenElement::Player => 5.0,
                            }/5.0)
                            .collect::<Vec<f64>>()
                    })
                    .collect::<Vec<Vec<f64>>>()
                    .concat();
                multi_seg.set_input(input_data.clone());
                multi_seg.next();
                let mut action: GameAction = GameAction::Hold;
                let mut last_value: f64 = -100.0;
                let output_data: Vec<f64> = multi_seg.get_output();
                if (output_data[0]).is_nan() {
                    panic!("NaN! Check if learning rate is to large.");
                }
                // for i in 0..output_data.len() {
                //     let value: f64 = output_data[i];
                //     if value > last_value {
                //         last_value = value;
                //         action = match i {
                //             0 => GameAction::Up,
                //             1 => GameAction::Down,
                //             2 => GameAction::Left,
                //             3 => GameAction::Right,
                //             _ => GameAction::Hold,
                //         };
                //     }
                // }
                action = random_as_probability::<GameAction>(
                    Vec::from([
                        GameAction::Up,
                        GameAction::Down,
                        GameAction::Left,
                        GameAction::Right,
                        GameAction::Hold,
                    ]),
                    output_data.clone(),
                );
                game.action(action);
                data_pair_list.push([input_data, output_data]);
            }
            let score: f64 = game.get_score();
            println!("score: {score}");
            let reword = score - 0.5;
            for data_pair in data_pair_list {
                multi_seg.set_input(data_pair[0].clone());
                multi_seg.next();
                multi_seg.operate_seg(flexible_net_id, |seg| {
                    seg.fitting(
                        data_pair[1]
                            .clone()
                            .iter()
                            .map(|v| v * reword)
                            .collect::<Vec<f64>>(),
                        rate,
                    );
                });
            }
        }

        let mut file = File::create("model/net.json").unwrap();
        let _ = file.write_all(multi_seg.export_data().as_bytes());

        println!("done!");
    }
    // until_ok!(try_fitting);
    for _ in 0..1000 {
        try_fitting();
    }

    match start_time.elapsed() {
        Ok(elapsed) => {
            println!("use time: {}ms", elapsed.as_millis());
        }
        Err(_) => {}
    }
}
