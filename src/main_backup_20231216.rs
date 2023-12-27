// new => init => get => set => other

use std::vec::Vec;
pub type Fsize = f64;

struct Node {
    is_input: bool, 
    input_count: usize, 
    input_value: Vec<Fsize>, 
    value: Fsize, 
    d: Fsize, 
    output_id: Vec<usize>, 
    output_index: Vec<usize>, 
    output_w: Vec<Fsize>
}
impl Node {
    pub fn new() -> Self {
        Node {
            is_input: false, 
            input_count: 0, 
            input_value: Vec::new(), 
            value: 0.0, 
            d: 0.0, 
            output_id: Vec::new(), 
            output_index: Vec::new(), 
            output_w: Vec::new()
        }
    }
    pub fn new_input() -> Self {
        Node {
            is_input: true, 
            input_count: 1, 
            input_value: Vec::new(), 
            value: 0.0, 
            d: 0.0, 
            output_id: Vec::new(), 
            output_index: Vec::new(), 
            output_w: Vec::new()
        }
    }
    pub fn new_input_source(&mut self) -> usize {
        self.input_count += 1;
        self.input_count - 1
    }
    pub fn new_output_target(&mut self, id: usize, index: usize, w: Fsize) {
        self.output_id.push(id);
        self.output_index.push(index);
        self.output_w.push(w);
    }
    pub fn init_input_value_vec(&mut self) {
        self.input_value.resize(self.input_count, 0.0);
    }
    pub fn get_value(&mut self) -> Fsize {
        self.value
    }
    pub fn calc_value(&mut self) -> Fsize {
        let mut value_sum: Fsize = 0.0;
        for value in self.input_value.iter() {
            value_sum += value;
        }
        self.value = value_sum + self.d;
        self.value
    }
    pub fn send_value(&mut self) -> Vec<Fsize> {
        let mut output_value = Vec::new();
        let _ = output_value.try_reserve(self.output_w.len());
        let _ = output_value.extend(self.output_w.iter().map(|&w| {
            w * self.value
        }));
        output_value
    }
}

struct Network {
    nodes: Vec<Node>, 
    input_id: Vec<usize>, 
    output_id: Vec<usize>
}
struct NetworkUpdateItem(usize, usize, Fsize);
impl Network {
    pub fn new() -> Self {
        Network {
            nodes: Vec::new(), 
            input_id: Vec::new(), 
            output_id: Vec::new()
        }
    }
    pub fn new_node(&mut self) -> usize {
        self.nodes.push(Node::new());
        self.nodes.len() - 1
    }
    pub fn new_input_node(&mut self) -> usize {
        self.nodes.push(Node::new_input());
        self.nodes.len() - 1
    }
    pub fn connect(&mut self, from_id: usize, to_id: usize, w: Fsize) {
        let to_index: usize = self.nodes[to_id].new_input_source();
        self.nodes[from_id].new_output_target(to_id, to_index, w);
    }
    pub fn init(&mut self) {
        for node in &mut self.nodes {
            node.init_input_value_vec();
        }
    }
    pub fn set_input_id(&mut self, input_id: Vec<usize>) {
        let _ = self.input_id.try_reserve(self.input_id.len());
        let _ = self.input_id.extend(input_id);
    }
    pub fn set_output_id(&mut self, output_id: Vec<usize>) {
        let _ = self.output_id.try_reserve(self.output_id.len());
        let _ = self.output_id.extend(output_id);
    }
    pub fn set_input(&mut self, input_value: Vec<Fsize>) {
        for i in 0..self.input_id.len() {
            let id: usize = self.input_id[i];
            let node: &mut Node = &mut self.nodes[id];
            if node.is_input {
                node.input_value[0] = input_value[i];
            } else {
                panic!("Node id {} is not an input node", id);
            }
        }
    }
    pub fn get_node(&mut self, id: usize) -> &mut Node {
        &mut self.nodes[id]
    }
    pub fn get_output(&mut self) -> Vec<Fsize> {
        let mut output_value: Vec<Fsize> = Vec::new();
        for i in 0..self.output_id.len() {
            let id: usize = self.output_id[i];
            let node: &mut Node = &mut self.nodes[id];
            output_value.push(node.get_value());
        }
        output_value
    }
    pub fn next(&mut self) {
        let mut update_data: Vec<NetworkUpdateItem> = Vec::new();
        for n in 0..self.nodes.len() {
            let node = self.get_node(n);
            node.calc_value();
            let output_value: Vec<Fsize> = node.send_value();
            for i in 0..output_value.len() {
                let id: usize = node.output_id[i];
                let index: usize = node.output_index[i];
                let value: Fsize = output_value[i];
                update_data.push(NetworkUpdateItem(id, index, value));
            }
        }
        for item in &update_data {
            self.get_node(item.0).input_value[item.1] = item.2;
        }
    }
}

fn main() {
    let mut net = Network::new();
    let a: usize = net.new_input_node();
    let b: usize = net.new_node();

    net.connect(a, b, 0.5);
    net.connect(b, b, 0.5);
    net.init();

    net.set_input_id(Vec::from([a]));
    net.set_output_id(Vec::from([b]));
    net.set_input(Vec::from([5.0]));

    for _ in 0..10 {
        net.next();
        println!("{}", serde_json::to_string(&net.get_output()).unwrap());
    }

    println!("done!");
}