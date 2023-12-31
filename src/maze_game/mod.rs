/*
 * 2023 (c) MaoHuPi
 * rust-rl/src/maze_game/mod.rs
 * Implement from Prim Maze Algorithm
 */

use rand::Rng;

enum WallPhase {
    Left, 
    Right, 
    Up, 
    Down
}
pub fn generate_maze(width: usize, height: usize) -> Vec<Vec<usize>> {
    let mut rng: rand::prelude::ThreadRng = rand::thread_rng();
    let size: usize = width*height;
    let mut cell_visited: Vec<Vec<bool>> = vec![vec![false; width]; height];
    let mut right_wall_exist: Vec<Vec<bool>> = vec![vec![true; width]; height];
    let mut bottom_wall_exist: Vec<Vec<bool>> = vec![vec![true; width]; height];
    let mut unvisit_cell: Vec<usize> = vec![0; size].iter().enumerate().map(|(_n, i)| *i).collect::<Vec<usize>>();
    let mut visited_hasneighbor_cell: Vec<usize> = vec![0; size].iter().enumerate().map(|(_n, i)| *i).collect::<Vec<usize>>();
    let mut cell_neighbor_count: Vec<Vec<usize>> = vec![vec![4; width]; height];
    for i in 0..size {
        unvisit_cell.push(i);
    }
    for w in 0..width{
        cell_neighbor_count[0][w] -= 1;
        cell_neighbor_count[height-1][w] -= 1;
    }
    for h in 0..height{
        cell_neighbor_count[h][0] -= 1;
        cell_neighbor_count[h][width-1] -= 1;
    }
    // cell_neighbor_count[0][1] -= 1;
    // cell_neighbor_count[1][0] -= 1;
    let mut pos: [usize; 2] = [0; 2];
    loop {
        cell_visited[pos[1]][pos[0]] = true;
        let target_value: usize = pos[1]*width + pos[0];
        match unvisit_cell.iter().position(|n: &usize| *n == target_value) {
            Some(target_index) => { unvisit_cell.remove(target_index); }, 
            None => {}
        }
        match visited_hasneighbor_cell.iter().position(|n: &usize| *n == target_value) {
            Some(target_index) => { visited_hasneighbor_cell.remove(target_index); }, 
            None => {
                for nth_neighbor in 0..4 {
                    let new_pos: [usize; 2] = [
                        pos[0]+match nth_neighbor{0 => 0, 2 => 2, _ => 1}, 
                        pos[1]+match nth_neighbor{1 => 0, 3 => 2, _ => 1}
                    ];
                    if new_pos[0] < 1 || new_pos[0] > width || new_pos[1] < 1 || new_pos[1] > height { continue; }
                    cell_neighbor_count[new_pos[1]-1][new_pos[0]-1] -= 1;
                }
            }
        }
        if cell_neighbor_count[pos[1]][pos[0]] > 0 {
            visited_hasneighbor_cell.push(target_value);
        }

        let mut wall_aside: Vec<WallPhase> = Vec::new();
        if pos[0] > 0 && !cell_visited[pos[1]][pos[0]-1] {
            wall_aside.push(WallPhase::Left);
        }
        if pos[0] < width-1 && !cell_visited[pos[1]][pos[0]+1] {
            wall_aside.push(WallPhase::Right);
        }
        if pos[1] > 0 && !cell_visited[pos[1]-1][pos[0]] {
            wall_aside.push(WallPhase::Up);
        }
        if pos[1] < height-1 && !cell_visited[pos[1]+1][pos[0]] {
            wall_aside.push(WallPhase::Down);
        }
        
        if wall_aside.len() == 0 {
            if unvisit_cell.len() == 0 { break; }
            let new_pos_num: usize = visited_hasneighbor_cell[rng.gen_range(0..visited_hasneighbor_cell.len())];
            pos[0] = new_pos_num%width;
            pos[1] = new_pos_num/width;
            continue;
        }

        match wall_aside[rng.gen_range(0..wall_aside.len())] {
            WallPhase::Left => {
                right_wall_exist[pos[1]][pos[0]-1] = false;
                pos[0] -= 1;
            }, 
            WallPhase::Right => {
                right_wall_exist[pos[1]][pos[0]] = false;
                pos[0] += 1;
            }, 
            WallPhase::Up => {
                bottom_wall_exist[pos[1]-1][pos[0]] = false;
                pos[1] -= 1;
            }, 
            WallPhase::Down => {
                bottom_wall_exist[pos[1]][pos[0]] = false;
                pos[1] += 1;
            }
        }
    }
    let maze_width: usize = width*2 + 1;
    let maze_height: usize = height*2 + 1;
    let mut maze: Vec<Vec<usize>> = vec![vec![8; maze_width]; maze_height];
    for w in 0..maze_width {
        for h in 0..maze_height {
            let i = h*width + w;
            maze[h][w] = 
            if w == 0 || h == 0 || w == maze_width-1 || h == maze_height-1 {
                1
            } else if w%2 == 0 {
                // Horizontal is wall.
                if h%2 == 0 {
                    // Vertical is wall.
                    1
                } else {
                    if right_wall_exist[(h-1)/2][w/2-1] {1} else {0}
                }
            } else {
                if h%2 == 0 {
                    // Vertical is wall.
                    if bottom_wall_exist[h/2-1][(w-1)/2] {1} else {0}
                } else {
                    0
                }
            }
        }
    }
    maze
}