/*
 * 2023 (c) MaoHuPi
 * rust-rl/src/maze_game/mod.rs
 * Implement from Prim Maze Algorithm
 */

use colored::Colorize;
use rand::Rng;
use std::cmp;
use std::time::SystemTime;

enum WallPhase {
    Left,
    Right,
    Up,
    Down,
}
#[derive(Clone, Copy, PartialEq)]
pub enum ScreenElement {
    Empty,
    Road,
    Wall,
    StartPoint,
    DestinationPoint,
    Player,
}
#[allow(unused)]
pub enum GameAction {
    Up,
    Down,
    Left,
    Right,
    Hold,
}
pub struct Game {
    screen_height: usize,
    screen_width: usize,
    maze_height: usize,
    maze_width: usize,
    maze: Vec<Vec<ScreenElement>>,
    player_pos: [usize; 2],
    start_pos: [usize; 2],
    destination_pos: [usize; 2],
    playing: bool,
    timer: SystemTime,
    step_count: usize,
    score: f64,
}
#[allow(unused)]
impl Game {
    pub fn new() -> Self {
        Self {
            screen_height: 0,
            screen_width: 0,
            maze_height: 0,
            maze_width: 0,
            maze: Vec::new(),
            player_pos: [0; 2],
            start_pos: [0; 2],
            destination_pos: [0; 2],
            playing: false,
            timer: SystemTime::now(),
            step_count: 0,
            score: 0.0,
        }
    }
    pub fn set_screen_size(self: &mut Self, hxw: [usize; 2]) {
        if self.playing {
            panic!();
        }
        self.screen_height = hxw[0];
        self.screen_width = hxw[1];
    }
    pub fn set_maze_size(self: &mut Self, hxw: [usize; 2]) {
        if self.playing {
            panic!();
        }
        self.maze_height = hxw[0];
        self.maze_width = hxw[1];
    }
    pub fn start(self: &mut Self) {
        let mut rng: rand::prelude::ThreadRng = rand::thread_rng();
        self.maze = Self::generate_maze(self.maze_height, self.maze_width);
        self.playing = true;
        self.start_pos = [
            rng.gen_range(0..self.maze_height) * 2 + 1,
            rng.gen_range(0..self.maze_width) * 2 + 1,
        ];
        loop {
            self.destination_pos = [
                rng.gen_range(0..self.maze_height) * 2 + 1,
                rng.gen_range(0..self.maze_width) * 2 + 1,
            ];
            if self.destination_pos[0] != self.start_pos[0]
                && self.destination_pos[1] != self.start_pos[1]
            {
                break;
            }
        }
        self.player_pos = self.start_pos.clone();
        self.timer = SystemTime::now();
    }
    pub fn stop(self: &mut Self) {
        match self.timer.elapsed() {
            Ok(elapsed) => {
                fn sub_abs(a: usize, b: usize) -> usize {
                    if a > b {
                        a - b
                    } else {
                        b - a
                    }
                }
                let distance: usize = sub_abs(self.start_pos[0], self.destination_pos[0])
                    + sub_abs(self.start_pos[1], self.destination_pos[1]);
                self.score = 2.0
                    - 2.0
                        / (1.0
                            + std::f64::consts::E
                                .powf(-(self.step_count as f64) / (distance as f64)));
            }
            Err(_) => {}
        }
        self.playing = false;
    }
    pub fn playing(self: &mut Self) -> bool {
        self.playing
    }
    pub fn action(self: &mut Self, action_type: GameAction) {
        if self.playing {
            match action_type {
                GameAction::Up => {
                    if self.player_pos[0] > 0
                        && self.maze[self.player_pos[0] - 1][self.player_pos[1]]
                            != ScreenElement::Wall
                    {
                        self.player_pos[0] -= 1;
                    }
                }
                GameAction::Down => {
                    if self.player_pos[0] < self.maze_height * 2
                        && self.maze[self.player_pos[0] + 1][self.player_pos[1]]
                            != ScreenElement::Wall
                    {
                        self.player_pos[0] += 1;
                    }
                }
                GameAction::Left => {
                    if self.player_pos[1] > 0
                        && self.maze[self.player_pos[0]][self.player_pos[1] - 1]
                            != ScreenElement::Wall
                    {
                        self.player_pos[1] -= 1;
                    }
                }
                GameAction::Right => {
                    if self.player_pos[1] < self.maze_width * 2
                        && self.maze[self.player_pos[0]][self.player_pos[1] + 1]
                            != ScreenElement::Wall
                    {
                        self.player_pos[1] += 1;
                    }
                }
                GameAction::Hold => {}
            }
            self.step_count += 1;
            if self.player_pos[0] == self.destination_pos[0]
                && self.player_pos[1] == self.destination_pos[1]
            {
                self.stop();
            }
        }
    }
    pub fn get_step_count(self: &Self) -> usize {
        self.step_count
    }
    pub fn get_score(self: &Self) -> f64 {
        self.score
    }
    pub fn get_screen(self: &mut Self) -> Vec<Vec<ScreenElement>> {
        if self.playing {
            let mut maze = self.maze.clone();
            maze[self.start_pos[0]][self.start_pos[1]] = ScreenElement::StartPoint;
            maze[self.destination_pos[0]][self.destination_pos[1]] =
                ScreenElement::DestinationPoint;
            maze[self.player_pos[0]][self.player_pos[1]] = ScreenElement::Player;

            let mut maze_distance: [usize; 4] = [0; 4];
            maze_distance[0] = self.player_pos[0];
            maze_distance[1] = self.maze_height * 2 + 1 - maze_distance[0];
            maze_distance[2] = self.player_pos[1];
            maze_distance[3] = self.maze_width * 2 + 1 - maze_distance[2];

            let mut screen_distance: [usize; 4] = [0; 4];
            [screen_distance[0], screen_distance[1]] = if self.screen_height % 2 == 0 {
                [self.screen_height / 2; 2]
            } else {
                [(self.screen_height - 1) / 2, (self.screen_height + 1) / 2]
            };
            [screen_distance[2], screen_distance[3]] = if self.screen_height % 2 == 0 {
                [self.screen_width / 2; 2]
            } else {
                [(self.screen_width - 1) / 2, (self.screen_width + 1) / 2]
            };

            fn a_gt_b_sub_else_0(a: usize, b: usize) -> usize {
                if a > b {
                    a - b
                } else {
                    0
                }
            }
            let mut cut_index: Vec<usize> = (0..4)
                .into_iter()
                .map(|i| a_gt_b_sub_else_0(maze_distance[i], screen_distance[i]))
                .collect::<Vec<usize>>();
            cut_index[1] = self.maze_height * 2 + 1 - cut_index[1];
            cut_index[3] = self.maze_height * 2 + 1 - cut_index[3];
            maze = maze
                .iter()
                .map(|row| row[cut_index[2]..cut_index[3]].to_vec())
                .collect::<Vec<Vec<ScreenElement>>>()[cut_index[0]..cut_index[1]]
                .to_vec();
            let justify_length: Vec<usize> = (0..4)
                .into_iter()
                .map(|i| a_gt_b_sub_else_0(screen_distance[i], maze_distance[i]))
                .collect::<Vec<usize>>();
            let just_top: Vec<Vec<ScreenElement>> =
                vec![vec![ScreenElement::Empty; self.screen_width]; justify_length[0]];
            let just_bottom: Vec<Vec<ScreenElement>> =
                vec![vec![ScreenElement::Empty; self.screen_width]; justify_length[1]];
            let just_left: Vec<ScreenElement> = vec![ScreenElement::Empty; justify_length[2]];
            let just_right: Vec<ScreenElement> = vec![ScreenElement::Empty; justify_length[3]];
            maze = maze
                .iter()
                .map(|row| {
                    Vec::from([just_left.clone(), row.to_vec(), just_right.clone()]).concat()
                })
                .collect::<Vec<Vec<ScreenElement>>>();
            maze = Vec::from([just_top, maze, just_bottom]).concat();
            maze
        } else {
            vec![vec![ScreenElement::Empty; self.screen_width]; self.screen_height]
        }
    }
    pub fn get_screen_string(self: &mut Self) -> String {
        self.get_screen()
            .iter()
            .map(|line| {
                line.iter()
                    .map(|&value| {
                        match value {
                            ScreenElement::Empty => "█".black(),
                            ScreenElement::Road => "█".black(),
                            ScreenElement::Wall => "█".yellow(),
                            ScreenElement::StartPoint => "█".green(),
                            ScreenElement::DestinationPoint => "█".red(),
                            ScreenElement::Player => "█".blue(),
                        }
                        .to_string()
                    })
                    .collect::<Vec<String>>()
                    .join(" ")
            })
            .collect::<Vec<String>>()
            .join("\n")
    }
    fn generate_maze(height: usize, width: usize) -> Vec<Vec<ScreenElement>> {
        let mut rng: rand::prelude::ThreadRng = rand::thread_rng();
        let size: usize = width * height;
        let mut cell_visited: Vec<Vec<bool>> = vec![vec![false; width]; height];
        let mut right_wall_exist: Vec<Vec<bool>> = vec![vec![true; width]; height];
        let mut bottom_wall_exist: Vec<Vec<bool>> = vec![vec![true; width]; height];
        let mut unvisit_cell: Vec<usize> = vec![0; size]
            .iter()
            .enumerate()
            .map(|(_n, i)| *i)
            .collect::<Vec<usize>>();
        let mut visited_hasneighbor_cell: Vec<usize> = vec![0; size]
            .iter()
            .enumerate()
            .map(|(_n, i)| *i)
            .collect::<Vec<usize>>();
        let mut cell_neighbor_count: Vec<Vec<usize>> = vec![vec![4; width]; height];
        for i in 0..size {
            unvisit_cell.push(i);
        }
        for w in 0..width {
            cell_neighbor_count[0][w] -= 1;
            cell_neighbor_count[height - 1][w] -= 1;
        }
        for h in 0..height {
            cell_neighbor_count[h][0] -= 1;
            cell_neighbor_count[h][width - 1] -= 1;
        }
        let mut pos: [usize; 2] = [0; 2];
        loop {
            cell_visited[pos[1]][pos[0]] = true;
            let target_value: usize = pos[1] * width + pos[0];
            match unvisit_cell.iter().position(|n: &usize| *n == target_value) {
                Some(target_index) => {
                    unvisit_cell.remove(target_index);
                }
                None => {}
            }
            match visited_hasneighbor_cell
                .iter()
                .position(|n: &usize| *n == target_value)
            {
                Some(target_index) => {
                    visited_hasneighbor_cell.remove(target_index);
                }
                None => {
                    for nth_neighbor in 0..4 {
                        let new_pos: [usize; 2] = [
                            pos[0]
                                + match nth_neighbor {
                                    0 => 0,
                                    2 => 2,
                                    _ => 1,
                                },
                            pos[1]
                                + match nth_neighbor {
                                    1 => 0,
                                    3 => 2,
                                    _ => 1,
                                },
                        ];
                        if new_pos[0] < 1
                            || new_pos[0] > width
                            || new_pos[1] < 1
                            || new_pos[1] > height
                        {
                            continue;
                        }
                        cell_neighbor_count[new_pos[1] - 1][new_pos[0] - 1] -= 1;
                    }
                }
            }
            if cell_neighbor_count[pos[1]][pos[0]] > 0 {
                visited_hasneighbor_cell.push(target_value);
            }

            let mut wall_aside: Vec<WallPhase> = Vec::new();
            if pos[0] > 0 && !cell_visited[pos[1]][pos[0] - 1] {
                wall_aside.push(WallPhase::Left);
            }
            if pos[0] < width - 1 && !cell_visited[pos[1]][pos[0] + 1] {
                wall_aside.push(WallPhase::Right);
            }
            if pos[1] > 0 && !cell_visited[pos[1] - 1][pos[0]] {
                wall_aside.push(WallPhase::Up);
            }
            if pos[1] < height - 1 && !cell_visited[pos[1] + 1][pos[0]] {
                wall_aside.push(WallPhase::Down);
            }

            if wall_aside.len() == 0 {
                if unvisit_cell.len() == 0 {
                    break;
                }
                let new_pos_num: usize =
                    visited_hasneighbor_cell[rng.gen_range(0..visited_hasneighbor_cell.len())];
                pos[0] = new_pos_num % width;
                pos[1] = new_pos_num / width;
                continue;
            }

            match wall_aside[rng.gen_range(0..wall_aside.len())] {
                WallPhase::Left => {
                    right_wall_exist[pos[1]][pos[0] - 1] = false;
                    pos[0] -= 1;
                }
                WallPhase::Right => {
                    right_wall_exist[pos[1]][pos[0]] = false;
                    pos[0] += 1;
                }
                WallPhase::Up => {
                    bottom_wall_exist[pos[1] - 1][pos[0]] = false;
                    pos[1] -= 1;
                }
                WallPhase::Down => {
                    bottom_wall_exist[pos[1]][pos[0]] = false;
                    pos[1] += 1;
                }
            }
        }
        let maze_width: usize = width * 2 + 1;
        let maze_height: usize = height * 2 + 1;
        let mut maze: Vec<Vec<ScreenElement>> =
            vec![vec![ScreenElement::Road; maze_width]; maze_height];
        for w in 0..maze_width {
            for h in 0..maze_height {
                let i = h * width + w;
                maze[h][w] = if w == 0 || h == 0 || w == maze_width - 1 || h == maze_height - 1 {
                    ScreenElement::Wall
                } else if w % 2 == 0 {
                    // Horizontal is wall.
                    if h % 2 == 0 {
                        // Vertical is wall.
                        ScreenElement::Wall
                    } else {
                        if right_wall_exist[(h - 1) / 2][w / 2 - 1] {
                            ScreenElement::Wall
                        } else {
                            ScreenElement::Road
                        }
                    }
                } else {
                    if h % 2 == 0 {
                        // Vertical is wall.
                        if bottom_wall_exist[h / 2 - 1][(w - 1) / 2] {
                            ScreenElement::Wall
                        } else {
                            ScreenElement::Road
                        }
                    } else {
                        ScreenElement::Road
                    }
                }
            }
        }
        maze
    }
}
