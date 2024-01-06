/*
 * 2024 (c) MaoHuPi
 * rust-rl/src/multi_seg_network/flexible_network.rs
 */

use std::collections::HashMap;
// HashMap Type
use std::f64::INFINITY;
// the Infinity Value
use serde::{Deserialize, Serialize};
// Make the customize struct be able to json stringify
use colored::Colorize;
// Colored print and panic.

use crate::multi_seg_network::{Segment, SegmentTypes};

fn check_ian(x: f64, message: String) {
    if x.is_infinite() {
        panic!(
            r#"[{}]: function input is Inf! Message: "{}"."#,
            "check_ian".red(),
            message.yellow()
        );
    } else if x.is_nan() {
        panic!(
            r#"[{}]: function input is NaN! Message: "{}"."#,
            "check_ian".red(),
            message.yellow()
        );
    }
}
fn check_domain(x: f64, min: f64, max: f64) -> Result<f64, String> {
    // To check the input value is between the min and max or not.
    if x > max {
        Err(format!(
            r#"[{}]: Function input out of range! x must lower then max value({}), but get "{}"."#,
            "check_domain".red(),
            max.to_string().yellow(),
            x.to_string().yellow()
        ))
    } else if x < min {
        Err(format!(
            r#"[{}]: Function input out of range! x must higher then min value({}), but get "{}"."#,
            "check_domain".red(),
            min.to_string().yellow(),
            x.to_string().yellow()
        ))
    } else {
        Ok(x)
    }
}

pub struct ActivationFunction();
#[derive(Clone, Copy)]
#[allow(dead_code)]
#[derive(Serialize, Deserialize)]
pub enum ActivationFunctionEnum {
    DoNothing,
    Sigmoid,
    Tanh,
    ReLU,
}
#[allow(dead_code)]
impl ActivationFunction {
    fn do_nothing(x: f64) -> f64 {
        x
    }
    fn do_nothing_derivative(x: f64) -> f64 {
        1.0
    }
    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + std::f64::consts::E.powf(-x))
    }
    fn sigmoid_inverse(x: f64) -> f64 {
        match check_domain(x, 0.0, 1.0) {
            Ok(v) => (v / (1.0 - v)).ln(),
            Err(e) => panic!("{}", e),
        }
    }
    fn sigmoid_derivative(x: f64) -> f64 {
        x / (1.0 - x)
    }
    fn tanh(x: f64) -> f64 {
        let e_powf_2x: f64 = std::f64::consts::E.powf(2.0 * x);
        (e_powf_2x - 1.0) / (e_powf_2x + 1.0)
    }
    fn tanh_inverse(x: f64) -> f64 {
        match check_domain(x, -1.0, 1.0) {
            Ok(v) => 0.5 * ((1.0 + v) / (1.0 - v)).ln(),
            Err(e) => panic!("{}", e),
        }
    }
    fn tanh_derivative(x: f64) -> f64 {
        1.0 - ActivationFunction::tanh(x).powi(2)
    }
    fn relu(x: f64) -> f64 {
        if x <= 0.0 {
            0.0
        } else {
            x
        }
    }
    fn relu_inverse(x: f64) -> f64 {
        match check_domain(x, 0.0, INFINITY) {
            Ok(v) => {
                if v <= 0.0 {
                    0.0
                } else {
                    v
                }
            }
            Err(e) => panic!("{}", e),
        }
    }
    fn relu_derivative(x: f64) -> f64 {
        if x <= 0.0 {
            0.0
        } else {
            1.0
        }
    }
    pub fn get_function(activation_fn_enum: ActivationFunctionEnum) -> fn(f64) -> f64 {
        match activation_fn_enum {
            ActivationFunctionEnum::DoNothing => ActivationFunction::do_nothing,
            ActivationFunctionEnum::Sigmoid => ActivationFunction::sigmoid,
            ActivationFunctionEnum::Tanh => ActivationFunction::tanh,
            ActivationFunctionEnum::ReLU => ActivationFunction::relu,
        }
    }
    pub fn get_inverse(activation_fn_enum: ActivationFunctionEnum) -> fn(f64) -> f64 {
        match activation_fn_enum {
            ActivationFunctionEnum::DoNothing => ActivationFunction::do_nothing,
            ActivationFunctionEnum::Sigmoid => ActivationFunction::sigmoid_inverse,
            ActivationFunctionEnum::Tanh => ActivationFunction::tanh_inverse,
            ActivationFunctionEnum::ReLU => ActivationFunction::relu_inverse,
        }
    }
    pub fn get_derivative(activation_fn_enum: ActivationFunctionEnum) -> fn(f64) -> f64 {
        match activation_fn_enum {
            ActivationFunctionEnum::DoNothing => ActivationFunction::do_nothing_derivative,
            ActivationFunctionEnum::Sigmoid => ActivationFunction::sigmoid_derivative,
            ActivationFunctionEnum::Tanh => ActivationFunction::tanh_derivative,
            ActivationFunctionEnum::ReLU => ActivationFunction::relu_derivative,
        }
    }
}

pub struct Node {
    // FlexibleNetwork Node
    id: usize,
    input_count: usize,
    input_id: Vec<usize>,
    input_w: Vec<f64>,
    // Weights of the pipe between this node and it's source node.
    input_value: Vec<f64>,
    input_partial: Vec<f64>,
    // $\frac{\partial self.input_value}{\partial lost}$
    output_count: usize,
    output_id: Vec<usize>,
    output_index: Vec<usize>,
    value: f64,
    anticipated_value: f64,
    partial: f64,
    // $\frac{\partial self.value}{\partial lost}$
    calc_planned: bool,
    // Operate by the "next" function of network.
    b: f64,
    activation_fn_enum: ActivationFunctionEnum, // Bias Terms
}
#[derive(Clone, Serialize, Deserialize)]
struct NodeData {
    id: usize,
    i_id: Vec<usize>,
    i_w: Vec<f64>,
    b: f64,
    a_fn: ActivationFunctionEnum,
}
pub struct NodeFetchQueueItem {
    // Queue Item for FlexibleNetwork Node to Fetch Value
    from_id: usize,
    to_id: usize,
    to_index: usize,
}
// When a node request value from it's source, it will return a list of "NodeFetchQueueItem".
// And the queue items can let the net work to update nodes' input value list.
#[allow(dead_code)]
impl Node {
    // FlexibleNetwork Node's Implementation
    pub fn new(id: usize) -> Self {
        // To create a new network node.
        Node {
            id,
            input_count: 0,
            input_id: Vec::new(),
            input_w: Vec::new(),
            input_value: Vec::new(),
            input_partial: Vec::new(),
            output_count: 0,
            output_id: Vec::new(),
            output_index: Vec::new(),
            value: 0.0,
            anticipated_value: 0.0,
            partial: 0.0,
            calc_planned: false,
            b: 0.0,
            activation_fn_enum: ActivationFunctionEnum::DoNothing,
        }
    }
    pub fn new_input_source(self: &mut Self, id: usize, w: f64) -> usize {
        // To write the new input source node's information(id, w, value's storage space).
        self.input_count += 1;
        self.input_id.push(id);
        self.input_w.push(w);
        self.input_value.push(0.0);
        self.input_partial.push(0.0);
        self.input_id.len() - 1
    }
    pub fn new_output_target(self: &mut Self, id: usize, index: usize) {
        // To write the new output target node's id.
        self.output_count += 1;
        self.output_id.push(id);
        self.output_index.push(index);
    }
    pub fn get_value(self: &mut Self) -> f64 {
        // Return the node's current value.
        self.value
    }
    pub fn get_partial(self: &mut Self, index: usize) -> f64 {
        // Return the node's current value.
        self.input_partial[index]
    }
    pub fn calc_value(self: &mut Self) -> f64 {
        // To calculate the node's value from add up all of the input value after multiply with the weights, plus bias terms, and pass through it's activation function.
        if self.input_count > 0 {
            let mut value_sum: f64 = 0.0;
            for i in 0..self.input_count {
                value_sum += self.input_value[i] * self.input_w[i];
                // self.input_value[i] = 0.0;
            }
            self.value = value_sum + self.b;
            self.value =
                ActivationFunction::get_function(self.activation_fn_enum.clone())(self.value);
        }
        self.value
    }
    pub fn fitting(self: &mut Self, learning_rate: f64, count_lost: bool) {
        if self.input_count > 0 {
            // let anticipated_value: f64 = ActivationFunction::get_inverse(self.activation_fn_enum.clone())(anticipated_value);
            // reverse anticipated_value by reversed activation function.

            /* 20240101 Update about Gradient Descent
             * rust-rl/DEVELOP.md > Develop Notes > 20240101 - A000001
             */
            /* 20240102 Question about Gradient Descent
             * rust-rl/DEVELOP.md > Develop Notes > 20240102 - A000002
             * rust-rl/DEVELOP.md > Develop Notes > 20240105 - A000003
             */
            // /* before 20240101 */cost := (self.value - inverse_activation_fn(anticipated_value)).powi(2);
            // cost := (self.value - anticipated_value).powi(2);
            let derivative_c_b: f64 = if count_lost {
                (2.0 * self.value - 2.0 * self.anticipated_value)
                    * (ActivationFunction::get_derivative(self.activation_fn_enum)(self.value))
            } else {
                (self.partial)
                    * (ActivationFunction::get_derivative(self.activation_fn_enum)(self.value))
            };

            let mut new_w: Vec<f64> = self.input_w.clone();
            for i in 0..self.input_count {
                let value_i = self.input_value[i];
                // /* before 20240101 */let gradient: f64 = 2.0*value_i.powi(2)*w_i + 2.0*(self.value-value_i*w_i + self.b - anticipated_value)*value_i;
                let gradient: f64 = derivative_c_b * (value_i);
                // check_ian(gradient, format!("2.0*{value_i}.powi(2)*{w_i} + 2.0*({0}-{1}*{w_i} + {2} - {anticipated_value})*{value_i}", self.value, value_i, self.b).to_string());
                new_w[i] -= gradient * learning_rate;
            }

            let mut new_b: f64 = self.b;
            // /* before 20240101 */let gradient: f64 = 2.0*(anticipated_value - self.value);
            let gradient: f64 = derivative_c_b * (1.0);
            new_b -= gradient * learning_rate;

            for i in 0..self.input_count {
                let w_i = self.input_w[i];
                // /* before 20240101 */let gradient: f64 = 2.0*w_i.powi(2)*value_i + 2.0*(self.value-value_i*w_i + self.b - anticipated_value)*w_i;
                let gradient: f64 = derivative_c_b * (w_i);
                self.input_partial[i] = gradient;
            }

            self.input_w = new_w;
            self.b = new_b;
        }
    }
    pub fn fetch_value(self: &mut Self) -> Vec<NodeFetchQueueItem> {
        // Return the list of "NodeFetchQueueItem", and queue to fetch value.
        self.input_id
            .iter()
            .enumerate()
            .map(|(index, id)| NodeFetchQueueItem {
                from_id: *id,
                to_id: self.id,
                to_index: index,
            })
            .collect::<Vec<NodeFetchQueueItem>>()
    }
    pub fn fetch_partial(self: &mut Self) -> Vec<NodeFetchQueueItem> {
        // Return the list of "NodeFetchQueueItem", and queue to fetch value.
        let mut queue_list: Vec<NodeFetchQueueItem> = Vec::new();
        for i in 0..self.output_count {
            queue_list.push(NodeFetchQueueItem {
                from_id: self.output_id[i],
                to_id: self.id,
                to_index: self.output_index[i],
            });
        }
        queue_list
    }
}

pub struct FlexibleNetwork {
    // FlexibleNetwork
    nodes: Vec<Node>,
    input_id: Vec<usize>,
    output_id: Vec<usize>,
    layer_length: HashMap<usize, usize>,
}
#[derive(Clone, Serialize, Deserialize)]
pub struct FlexibleNetworkData {
    ns: Vec<NodeData>,
    i_id: Vec<usize>,
    o_id: Vec<usize>,
    l_len: HashMap<usize, usize>,
}
impl FlexibleNetworkData {
    fn new() -> Self {
        Self {
            ns: Vec::new(),
            i_id: Vec::new(),
            o_id: Vec::new(),
            l_len: HashMap::new(),
        }
    }
}
struct FlexibleNetworkUpdateItem {
    // Item to Update FlexibleNetwork Node
    to_id: usize,
    to_index: usize,
    new_value: f64,
}
#[allow(dead_code)]
impl Segment for FlexibleNetwork {
    fn new() -> Self {
        // Create new network.
        Self {
            nodes: Vec::new(),
            input_id: Vec::new(),
            output_id: Vec::new(),
            layer_length: HashMap::new(),
        }
    }
    fn get_type(self: &Self) -> SegmentTypes {
        SegmentTypes::FlexibleNetwork
    }
    fn set_input(self: &mut Self, input_value: Vec<f64>) {
        // Set input values.
        for i in 0..self.input_id.len() {
            let node: &mut Node = &mut self.nodes[self.input_id[i]];
            node.value = input_value[i];
        }
    }
    fn get_output(self: &mut Self) -> Vec<f64> {
        // Return output value list.
        let mut output_value: Vec<f64> = Vec::new();
        for i in 0..self.output_id.len() {
            let id: usize = self.output_id[i];
            let node: &mut Node = &mut self.nodes[id];
            output_value.push(node.get_value());
        }
        output_value
    }
    fn export_data(self: &mut Self) -> String {
        let mut node_data_array: Vec<NodeData> = Vec::new();
        for id in 0..self.nodes.len() {
            let node: &mut Node = self.get_node(id);
            let node_data = NodeData {
                id,
                i_id: node.input_id.clone(),
                i_w: node.input_w.clone(),
                b: node.b,
                a_fn: node.activation_fn_enum,
            };
            node_data_array.push(node_data);
        }
        let data: FlexibleNetworkData = FlexibleNetworkData {
            ns: node_data_array,
            i_id: self.input_id.clone(),
            o_id: self.output_id.clone(),
            l_len: self.layer_length.clone(),
        };
        serde_json::to_string(&data).unwrap()
    }
    fn import_data(self: &mut Self, data: String) {
        let data: FlexibleNetworkData = serde_json::from_str(&data.as_str()).unwrap();
        for node_data in data.ns {
            let mut node: Node = Node::new(node_data.id);
            node.activation_fn_enum = node_data.a_fn;
            node.input_count = node_data.i_id.len();
            node.input_id = node_data.i_id.clone();
            node.input_w = node_data.i_w.clone();
            node.input_value = node.input_id.iter().map(|&x| 0.0).collect::<Vec<f64>>();
            self.nodes.push(node);
        }
        self.input_id = data.i_id;
        self.output_id = data.o_id;
        self.layer_length = data.l_len;
    }
    fn next(self: &mut Self) {
        // Next step of this network.
        // Update value from top to bottom in the node chain.

        for id in 0..self.nodes.len() {
            self.nodes[id].calc_planned = false;
        }
        // Set all of the node to not calculated yet.

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
            let mut update_data: Vec<FlexibleNetworkUpdateItem> = Vec::new();
            for queue_item in queue_list {
                let from_id: usize = queue_item.from_id;
                if self.nodes[from_id].input_count == 0 || from_id == queue_item.to_id {
                    update_data.push(FlexibleNetworkUpdateItem {
                        to_id: queue_item.to_id,
                        to_index: queue_item.to_index,
                        new_value: self.nodes[from_id].get_value(),
                    });
                } else {
                    still_queue_list.push(queue_item);
                    if !self.nodes[from_id].calc_planned {
                        // Use "calc_planned" flag to prevent a node chain fetch value so more times.
                        new_queue_list.append(&mut self.nodes[from_id].fetch_value());
                        self.nodes[from_id].calc_planned = true;
                    }
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
    fn can_fitting(self: &Self) -> bool {
        true
    }
    fn fitting(self: &mut Self, anticipated_output: Vec<f64>, learning_rate: f64) {
        for id in 0..self.nodes.len() {
            self.nodes[id].calc_planned = false;
        }
        // Set all of the node to not calculated yet.
        for id in 0..self.nodes.len() {
            self.nodes[id].partial = 0.0;
        }
        // Set all of the node value partial to zero.
        for i in 0..self.output_id.len() {
            self.nodes[self.output_id[i]].anticipated_value = anticipated_output[i];
            self.nodes[self.output_id[i]].fitting(learning_rate, true);
        }

        let mut queue_list: Vec<NodeFetchQueueItem> = Vec::new();
        let input_id: Vec<usize> = self.input_id.clone();
        for id in input_id {
            queue_list.append(&mut self.nodes[id].fetch_partial());
        }
        // Append all of the fetch queue item into queue list.

        let mut still_queue_list: Vec<NodeFetchQueueItem> = Vec::new();
        loop {
            if queue_list.len() == 0 {
                break;
            }
            let mut new_queue_list: Vec<NodeFetchQueueItem> = Vec::new();
            let mut update_data: Vec<FlexibleNetworkUpdateItem> = Vec::new();
            for queue_item in queue_list {
                let from_id: usize = queue_item.from_id;
                if self.nodes[from_id].output_count == 0 || from_id == queue_item.to_id {
                    update_data.push(FlexibleNetworkUpdateItem {
                        to_id: queue_item.to_id,
                        to_index: 0,
                        new_value: self.nodes[from_id].get_partial(queue_item.to_index),
                    });
                } else {
                    still_queue_list.push(queue_item);
                    if !self.nodes[from_id].calc_planned {
                        // Use "calc_planned" flag to prevent a node chain fetch value so more times.
                        new_queue_list.append(&mut self.nodes[from_id].fetch_partial());
                        self.nodes[from_id].calc_planned = true;
                    }
                }
            }
            for item in &update_data {
                self.get_node(item.to_id).partial += item.new_value;
            }
            queue_list = new_queue_list;
        }
        // To check if the source node is input node one by one, then set it's value back to the fetching node.

        for queue_item in still_queue_list.iter().rev() {
            self.nodes[queue_item.from_id].fitting(learning_rate, false);
            self.nodes[queue_item.to_id].partial +=
                self.nodes[queue_item.from_id].get_partial(queue_item.to_index);
        }
        // Calculate node's value and transfer it in reverse.
    }
}
#[allow(dead_code)]
impl FlexibleNetwork {
    // FlexibleNetwork's Implementation
    pub fn new_node(self: &mut Self, b: f64, activation_fn_enum: ActivationFunctionEnum) -> usize {
        // Create new node and initialize it.
        let id = self.nodes.len();
        let mut node = Node::new(id);
        node.b = b;
        node.activation_fn_enum = activation_fn_enum;
        self.nodes.push(node);
        id
    }
    pub fn new_layer(
        self: &mut Self,
        node_number: usize,
        b: f64,
        activation_fn_enum: ActivationFunctionEnum,
    ) -> usize {
        // Create new node and initialize it.
        let id_start = self.nodes.len();
        for i in 0..node_number {
            let mut node = Node::new(id_start + i);
            node.b = b;
            node.activation_fn_enum = activation_fn_enum;
            self.nodes.push(node);
        }
        self.layer_length.insert(id_start, node_number);
        id_start
    }
    pub fn connect(self: &mut Self, from_id: usize, to_id: usize, w: f64) {
        // Connect two of the node that is in this network.
        let index: usize = self.nodes[to_id].new_input_source(from_id, w);
        self.nodes[from_id].new_output_target(to_id, index);
    }
    // fn get_id_array_from_layer(self: &mut Self, layer_id: usize) -> Vec<usize> {
    //     match self.layer_length.get(&layer_id) {
    //         Some(layer_length) => (layer_id..layer_id+layer_length).collect::<Vec<usize>>(),
    //         None => panic!("[{}]: Layer not found!", "get_id_array_from_layer")
    //     }
    // }
    pub fn connect_layer(self: &mut Self, from_layer: usize, to_layer: usize, w: f64) {
        let from_layer_length: &usize = self.layer_length.get(&from_layer).unwrap();
        let to_layer_length: &usize = self.layer_length.get(&to_layer).unwrap();
        for f in from_layer..from_layer + from_layer_length {
            for t in to_layer..to_layer + to_layer_length {
                let index: usize = self.nodes[t].new_input_source(f, w);
                self.nodes[f].new_output_target(t, index);
            }
        }
    }
    pub fn set_input_id(self: &mut Self, input_id: Vec<usize>) {
        // Set the node ids corresponding to the input values.
        let _ = self.input_id.try_reserve(self.input_id.len());
        let _ = self.input_id.extend(input_id);
    }
    pub fn set_input_layer(self: &mut Self, input_layer: usize) {
        match self.layer_length.get(&input_layer) {
            Some(layer_length) => {
                self.input_id = (input_layer..input_layer + layer_length).collect::<Vec<usize>>()
            }
            None => panic!("[{}]: Layer not found!", "set_input_layer"),
        }
    }
    pub fn set_output_id(self: &mut Self, output_id: Vec<usize>) {
        // Set the node ids corresponding to the output values.
        let _ = self.output_id.try_reserve(self.output_id.len());
        let _ = self.output_id.extend(output_id);
    }
    pub fn set_output_layer(self: &mut Self, output_layer: usize) {
        match self.layer_length.get(&output_layer) {
            Some(layer_length) => {
                self.output_id = (output_layer..output_layer + layer_length).collect::<Vec<usize>>()
            }
            None => panic!("[{}]: Layer not found!", "set_output_layer"),
        }
    }
    pub fn get_node(self: &mut Self, id: usize) -> &mut Node {
        // Return nth node in this network's node list.
        &mut self.nodes[id]
    }
}

#[cfg(test)]
mod tests {
    use super::Segment;
    use super::{ActivationFunctionEnum, FlexibleNetwork};

    #[test]
    fn test_node_fitting() {
        let test_data: [[f64; 2]; 2] = [[1.0, 2.0], [2.0, 4.0]];
        let mut net = FlexibleNetwork::new();
        let i_id: usize = net.new_node(0.0, ActivationFunctionEnum::DoNothing);
        let o_id: usize = net.new_node(0.0, ActivationFunctionEnum::DoNothing);
        net.connect(i_id, o_id, 1.0);
        net.set_input_id(Vec::from([i_id]));
        net.set_output_id(Vec::from([o_id]));

        for rate in [0.001, 0.0001] {
            for _ in 1..=10000 {
                for test_pair in test_data {
                    net.set_input(Vec::from([test_pair[0]]));
                    net.next();
                    let node = net.get_node(o_id);
                    node.anticipated_value = test_pair[1];
                    node.fitting(rate, true);
                }
            }
        }

        for test_pair in test_data {
            net.set_input(Vec::from([test_pair[0]]));
            net.next();
            assert!((net.get_output()[0] - test_pair[1]).abs() < 1.0);
        }
    }

    #[test]
    fn test_network_fitting() {
        let test_data: [[f64; 3]; 3] = [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.5, 1.2, 2.7]];
        let mut net = FlexibleNetwork::new();
        let input_layer: usize = net.new_layer(2, 0.0, ActivationFunctionEnum::DoNothing);
        let hidden_layer: usize = net.new_layer(5, 0.0, ActivationFunctionEnum::DoNothing);
        let output_layer: usize = net.new_layer(1, 0.0, ActivationFunctionEnum::ReLU);
        net.connect_layer(input_layer, hidden_layer, 1.0);
        net.connect_layer(hidden_layer, output_layer, 1.0);
        net.set_input_layer(input_layer);
        net.set_output_layer(output_layer);

        for rate in [0.001, 0.0001] {
            for _ in 1..=10000 {
                for test_pair in test_data {
                    net.set_input(Vec::from(&test_pair[0..=1]));
                    net.next();
                    net.fitting(Vec::from([test_pair[2]]), rate);
                }
            }
        }

        for test_pair in test_data {
            net.set_input(Vec::from(&test_pair[0..=1]));
            net.next();
            assert!((net.get_output()[0] - test_pair[2]).abs() < 1.0);
        }
    }
}
