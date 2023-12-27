// 2023 (c) MaoHuPi
// rust-rl/src/flexible_network/network.rs

pub struct ActivationFunction();
#[allow(dead_code)]
impl ActivationFunction {
    pub fn do_nothing(x: f64) -> f64 {
        x
    }
    pub fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + std::f64::consts::E.powf(-x))
    }
    pub fn tanh(x: f64) -> f64 {
        let e_powf_2x: f64 = std::f64::consts::E.powf(2.0*x);
        (e_powf_2x - 1.0) / (e_powf_2x + 1.0)
    }
    pub fn relu(x: f64) -> f64 {
        if x <= 0.0 {0.0} else {x}
    }
    // pub fn softmax(x: Vec<f64>) -> Vec<f64> {
    //     Vec::new()
    // }
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
    activation_fn: fn(f64) -> f64
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
            activation_fn: ActivationFunction::do_nothing
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
            }
            self.value = value_sum + self.b;
            self.value = (self.activation_fn)(self.value);
        }
        self.value
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
    pub fn new_node(&mut self, b: f64, activation_fn: fn(f64) -> f64) -> usize {
        // Create new node and initialize it.
        let id = self.nodes.len();
        let mut node = Node::new(id);
        node.b = b;
        node.activation_fn = activation_fn;
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
}