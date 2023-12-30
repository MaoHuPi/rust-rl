// 2023 (c) MaoHuPi
// rust-rl/src/flexible_network/network.rs

pub struct ActivationFunction();
#[derive(Clone)]
#[allow(dead_code)]
pub enum ActivationFunctionEnum {
    DoNothing, 
    Sigmoid, 
    Tanh, 
    ReLU
}
#[allow(dead_code)]
impl ActivationFunction {
    fn do_nothing(x: f64) -> f64 {
        x
    }
    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + std::f64::consts::E.powf(-x))
    }
    fn sigmoid_inverse(x: f64) -> f64 {
        (x/(1.0-x)).ln()
    }
    fn tanh(x: f64) -> f64 {
        let e_powf_2x: f64 = std::f64::consts::E.powf(2.0*x);
        (e_powf_2x - 1.0) / (e_powf_2x + 1.0)
    }
    fn tanh_inverse(x: f64) -> f64 {
        0.5*((1.0+x)/(1.0-x)).ln()
    }
    fn relu(x: f64) -> f64 {
        if x <= 0.0 {0.0} else {x}
    }
    pub fn get_function(activation_fn_enum: ActivationFunctionEnum) -> fn(f64) -> f64 {
        match activation_fn_enum {
            ActivationFunctionEnum::DoNothing => ActivationFunction::do_nothing, 
            ActivationFunctionEnum::Sigmoid => ActivationFunction::sigmoid, 
            ActivationFunctionEnum::Tanh => ActivationFunction::tanh, 
            ActivationFunctionEnum::ReLU => ActivationFunction::relu
        }
    }
    pub fn get_inverse(activation_fn_enum: ActivationFunctionEnum) -> fn(f64) -> f64 {
        match activation_fn_enum {
            ActivationFunctionEnum::DoNothing => ActivationFunction::do_nothing, 
            ActivationFunctionEnum::Sigmoid => ActivationFunction::sigmoid_inverse, 
            ActivationFunctionEnum::Tanh => ActivationFunction::tanh_inverse, 
            ActivationFunctionEnum::ReLU => ActivationFunction::relu
        }
    }
}

pub struct Node {
    // Network Node
    id: usize, 
    input_count: usize, 
    input_id: Vec<usize>, 
    input_w: Vec<f64>, 
    // Weights of the pipe between this node and it's source node.
    input_value: Vec<f64>, 
    value: f64, 
    b: f64, 
    activation_fn_enum: ActivationFunctionEnum
    // Bias Terms
}
pub struct NodeFetchQueueItem {
    // Queue Item for Network Node to Fetch Value
    from_id: usize, 
    to_id: usize, 
    to_index: usize
}
// When a node request value from it's source, it will return a list of "NodeFetchQueueItem".
// And the queue items can let the net work to update nodes' input value list.
#[allow(dead_code)]
impl Node {
    // Network Node's Implementation
    pub fn new(id: usize) -> Self {
        // To create a new network node.
        Node {
            id, 
            input_count: 0, 
            input_id: Vec::new(), 
            input_w: Vec::new(), 
            input_value: Vec::new(), 
            value: 0.0, 
            b: 0.0, 
            activation_fn_enum: ActivationFunctionEnum::DoNothing
        }
    }
    pub fn new_input_source(&mut self, id: usize, w: f64) {
        // To write the new input source node's information(id, w, value's storage space).
        self.input_count += 1;
        self.input_id.push(id);
        self.input_w.push(w);
        self.input_value.push(0.0);
    }
    pub fn get_value(&mut self) -> f64 {
        // Return the node's current value.
        self.value
    }
    pub fn calc_value(&mut self) -> f64 {
        // To calculate the node's value from add up all of the input value after multiply with the weights, plus bias terms, and pass through it's activation function.
        if self.input_count > 0 {
            let mut value_sum: f64 = 0.0;
            for i in 0..self.input_count{
                value_sum += self.input_value[i] * self.input_w[i];
                // self.input_value[i] = 0.0;
            }
            self.value = value_sum + self.b;
            self.value = ActivationFunction::get_function(self.activation_fn_enum.clone())(self.value);
        }
        self.value
    }
    pub fn fitting(&mut self, anticipated_value: f64, learning_rate: f64) {
        if self.input_count > 0 {
            // reverse anticipated_value by activation_fn at first!
            let anticipated_value: f64 = ActivationFunction::get_inverse(self.activation_fn_enum.clone())(anticipated_value);
            
            // cost := (self.value - anticipated_value).powi(2);
            for i in 0..self.input_count {
                self.calc_value();
                let value_i = self.input_value[i];
                let w_i = self.input_w[i];
                let gradient: f64 = 2.0*value_i.powi(2)*w_i + 2.0*(self.value-value_i*w_i + self.b - anticipated_value)*value_i;
                // partial w_i of cost := 4.0*value_i.powi(2)*w_i + 2.0*(self.value-value_i*w_i + self.b - anticipated_value)*value_i;
                self.input_w[i] = w_i - gradient*learning_rate;
            }
            self.calc_value();
            let gradient: f64 = 2.0*self.b + 2.0*(self.value-self.b - anticipated_value);
            // partial b of cost := 4.0*self.b + 2.0*(self.value-self.b - anticipated_value);
            self.b = self.b - gradient*learning_rate;
            
            for i in 0..self.input_count {
                self.calc_value();
                let value_i = self.input_value[i];
                let w_i = self.input_w[i];
                let gradient: f64 = 2.0*w_i.powi(2)*value_i + 2.0*(self.value-value_i*w_i + self.b - anticipated_value)*w_i;
                // partial value_i of cost := 2.0*w_i.powi(2)*value_i + 2.0*(self.value-value_i*w_i + self.b - anticipated_value)*w_i;
                self.input_value[i] = value_i - gradient*learning_rate;
            }
        }
    }
    pub fn fetch_value(&mut self) -> Vec<NodeFetchQueueItem> {
        // Return the list of "NodeFetchQueueItem", and queue to fetch value.
        self.input_id.iter().enumerate().map(|(index, id)| NodeFetchQueueItem{from_id: *id, to_id: self.id, to_index: index}).collect::<Vec<NodeFetchQueueItem>>()
    }
}

pub struct Network {
    // Network
    nodes: Vec<Node>, 
    input_id: Vec<usize>, 
    output_id: Vec<usize>
}
struct NetworkUpdateItem{
    // Item to Update Network Node 
    to_id: usize, 
    to_index: usize, 
    new_value: f64
}
#[allow(dead_code)]
impl Network {
    // Network's Implementation
    pub fn new() -> Self {
        // Create new network.
        Network {
            nodes: Vec::new(), 
            input_id: Vec::new(), 
            output_id: Vec::new()
        }
    }
    pub fn new_node(&mut self, b: f64, activation_fn_enum: ActivationFunctionEnum) -> usize {
        // Create new node and initialize it.
        let id = self.nodes.len();
        let mut node = Node::new(id);
        node.b = b;
        node.activation_fn_enum = activation_fn_enum;
        self.nodes.push(node);
        id
    }
    pub fn connect(&mut self, from_id: usize, to_id: usize, w: f64) {
        // Connect two of the node that is in this network.
        self.nodes[to_id].new_input_source(from_id, w);
    }
    pub fn set_input_id(&mut self, input_id: Vec<usize>) {
        // Set the node ids corresponding to the input values. 
        let _ = self.input_id.try_reserve(self.input_id.len());
        let _ = self.input_id.extend(input_id);
    }
    pub fn set_output_id(&mut self, output_id: Vec<usize>) {
        // Set the node ids corresponding to the output values. 
        let _ = self.output_id.try_reserve(self.output_id.len());
        let _ = self.output_id.extend(output_id);
    }
    pub fn set_input(&mut self, input_value: Vec<f64>) {
        // Set input values.
        for i in 0..self.input_id.len() {
            let node: &mut Node = &mut self.nodes[self.input_id[i]];
            node.value = input_value[i];
        }
    }
    pub fn get_node(&mut self, id: usize) -> &mut Node {
        // Return nth node in this network's node list.
        &mut self.nodes[id]
    }
    pub fn get_output(&mut self) -> Vec<f64> {
        // Return output value list.
        let mut output_value: Vec<f64> = Vec::new();
        for i in 0..self.output_id.len() {
            let id: usize = self.output_id[i];
            let node: &mut Node = &mut self.nodes[id];
            output_value.push(node.get_value());
        }
        output_value
    }
    pub fn next(&mut self) {
        // Next step of this network.
        // Update value from top to bottom in the node chain. 
        let mut queue_list: Vec<NodeFetchQueueItem> = Vec::new();
        let output_id: Vec<usize> = self.output_id.clone();
        for id in output_id {
            queue_list.append(&mut self.nodes[id].fetch_value());
        }
        // Append all of the fetch queue item into queue list.

        let mut still_queue_list: Vec<NodeFetchQueueItem> = Vec::new();
        loop {
            if queue_list.len() == 0 {
                break;
            }
            let mut new_queue_list: Vec<NodeFetchQueueItem> = Vec::new();
            let mut update_data: Vec<NetworkUpdateItem> = Vec::new();
            for queue_item in queue_list {
                let from_id: usize = queue_item.from_id;
                if self.nodes[from_id].input_count == 0 || from_id == queue_item.to_id {
                    update_data.push(NetworkUpdateItem{
                        to_id: queue_item.to_id, 
                        to_index: queue_item.to_index, 
                        new_value: self.nodes[from_id].get_value()
                    });
                } else {
                    still_queue_list.push(queue_item);
                    new_queue_list.append(&mut self.nodes[from_id].fetch_value());
                }
            }
            for item in &update_data {
                self.get_node(item.to_id).input_value[item.to_index] = item.new_value;
            }
            queue_list = new_queue_list;
        }
        // To check if the source node is input node one by one, then set it's value back to the fetching node.

        for queue_item in still_queue_list.iter().rev() {
            let value: f64 = self.nodes[queue_item.from_id].calc_value();
            self.nodes[queue_item.to_id].input_value[queue_item.to_index] = value;
        }
        // Calculate node's value and transfer it in reverse.

        let output_id: Vec<usize> = self.output_id.clone();
        for id in output_id {
            self.nodes[id].calc_value();
        }
        // Calculate output nodes' value.
    }
    pub fn fitting(&mut self, anticipated_output: Vec<f64>, learning_rate: f64) {
        let mut queue_list: Vec<NodeFetchQueueItem> = Vec::new();
        for i in 0..self.output_id.len() {
            self.nodes[self.output_id[i]].fitting(anticipated_output[i], learning_rate);
            queue_list.append(&mut self.nodes[self.output_id[i]].fetch_value());
        }

        loop {
            if queue_list.len() == 0 {
                break;
            }
            let mut new_queue_list: Vec<NodeFetchQueueItem> = Vec::new();
            for queue_item in queue_list {
                let from_id: usize = queue_item.from_id;
                if !(self.nodes[from_id].input_count == 0 || from_id == queue_item.to_id) {
                    let anticipated_input_value: f64 = self.nodes[queue_item.to_id].input_value[queue_item.to_index];
                    self.nodes[from_id].fitting(anticipated_input_value, learning_rate);
                    new_queue_list.append(&mut self.nodes[from_id].fetch_value());
                }
            }
            queue_list = new_queue_list;
        }
        // some times the vector value exhibits NaN!
    }
}

#[cfg(test)]
mod tests {
    use super::{Network, ActivationFunctionEnum};
    use std::convert::TryFrom;
    use rand::Rng;

    #[test]
    fn test_node_fitting() {
        let mut net = Network::new();
        let i_id: usize = net.new_node(0.0, ActivationFunctionEnum::DoNothing);
        let o_id: usize = net.new_node(0.0, ActivationFunctionEnum::DoNothing);
        net.connect(i_id, o_id, 1.0);
        net.set_input_id(Vec::from([i_id]));
        net.set_output_id(Vec::from([o_id]));
        
        macro_rules! setup_and_fitting {
            ($input: expr => $output: expr, $rate: expr) => {
                {
                    for n in 0..5{
                        net.set_input(Vec::from([$input*(n as f64)]));
                        net.next();
                        net.get_node(o_id).fitting($output*(n as f64), $rate);
                    }
                }
            }
        }
        for _ in 1..=100 { setup_and_fitting!(1.0 => 2.0, 0.001); }
        for i in 1..=10 { setup_and_fitting!(1.0 => 2.0, 0.001/(f64::try_from(i).unwrap())); }
        for _ in 1..=1000 { setup_and_fitting!(1.0 => 2.0, 0.000001); }
        
        net.set_input(Vec::from([10.0]));
        net.next();
        // let o = net.get_node(o_id);
        // let o_b = o.b;
        // println!("{} {}", o.input_w[0], o_b);
        // assert_eq!(net.get_output(), Vec::from([10.0]));
        assert_eq!(net.get_output(), Vec::from([18.143440974766087]));
    }
    
    // #[test]
    // fn test_network_fitting() {
    //     let mut rng = rand::thread_rng();
    //     let mut net = Network::new();
    //     let mut input_layer: Vec<usize> = Vec::new();
    //     let mut hidden_layer: Vec<usize> = Vec::new();
    //     let mut output_layer: Vec<usize> = Vec::new();
    //     for _ in 0..10 {
    //         input_layer.push(net.new_node(0.0, ActivationFunctionEnum::ReLU));
    //     }
    //     for _ in 0..10 {
    //         let h_id: usize = net.new_node(0.0, ActivationFunctionEnum::ReLU);
    //         for i_id in input_layer.iter() {
    //             net.connect(*i_id, h_id, rng.gen_range(-1.0..1.0));
    //         }
    //         if h_id%2 == 0{
    //             net.connect(h_id, h_id, rng.gen_range(-1.0..1.0));
    //         }
    //         hidden_layer.push(h_id);
    //     }
    //     for _ in 0..3 {
    //         let o_id: usize = net.new_node(0.0, ActivationFunctionEnum::ReLU);
    //         for h_id in hidden_layer.iter() {
    //             net.connect(*h_id, o_id, rng.gen_range(0.0..1.0));
    //         }
    //         output_layer.push(o_id);
    //     }
    //     net.set_input_id(input_layer);
    //     net.set_output_id(output_layer);
        
    //     macro_rules! setup_and_fitting {
    //         ($input: expr => $output: expr, $rate: expr) => {
    //             {
    //                 for n in 0..5{
    //                     net.set_input(Vec::from([$input*(n as f64)]));
    //                     net.next();
    //                     net.fitting(Vec::from([$output*(n as f64)]), $rate);
    //                 }
    //             }
    //         }
    //     }
    //     for _ in 1..=100 { setup_and_fitting!(1.0 => 2.0, 0.001); }
    //     for i in 1..=10 { setup_and_fitting!(1.0 => 2.0, 0.001/(f64::try_from(i).unwrap())); }
    //     for _ in 1..=1000 { setup_and_fitting!(1.0 => 2.0, 0.000001); }
        
    //     net.set_input(Vec::from([10.0]));
    //     net.next();
    //     let o = net.get_node(o_id);
    //     let o_b = o.b;
    //     println!("{} {}", o.input_w[0], o_b);
    //     assert_eq!(net.get_output(), Vec::from([10.0]));
    //     // assert_eq!(net.get_output(), Vec::from([18.143440974766087]));
    // }
}