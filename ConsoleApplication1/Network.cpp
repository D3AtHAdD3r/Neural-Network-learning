#include "network.hpp"
#include <iomanip>
#include"utils.h"


//Initializes the network with layers defined by sizes(e.g., {784, 30, 10}).
//Stores the number of layers and sizes.
//MNIST Context : For[784, 30, 10], the network takes a 784 - pixel image, processes it through 30 hidden neurons, and outputs 10 scores(one per digit).
Network::Network(const std::vector<int>& sizes) : sizes(sizes), num_layers(sizes.size()), rng(std::random_device{}()) {
    
    // Initialize layers: each layer connects sizes[i] to sizes[i+1]
    for (size_t i = 1; i < sizes.size(); ++i) {
        layers.emplace_back(sizes[i - 1], sizes[i], static_cast<unsigned int>(rng()));
    }
}

//Computes the network’s output for an input vector
//Input a is a 784x1 vector (MNIST image pixels, normalized to [0,1]).
//For each layer:
//Compute Activation
//Pass the new activation to the next layer.
//MNIST Context: Output is a 10x1 vector, where the highest value’s index is the predicted digit.
//MNIST Context: An image like 5 might produce outputs like [0.1, 0.05, 0.1, 0.05, 0.05, 0.6, 0.05, 0.0, 0.0, 0.0], predicting 5.
Eigen::VectorXd Network::feedforward(const Eigen::VectorXd& a) {
   
    Eigen::VectorXd activation = a;
    for (auto& layer : layers) {
        activation = layer.forward(activation);
    }
    return activation;
}


//Trains the network using mini-batch SGD.
//Parameters:
//training_data: Vector of (input, label) pairs (input: 784x1, label: 10x1 one-hot).
//epochs: Number of full passes over the training data.
//mini_batch_size: Number of examples per mini-batch (e.g., 10).
//eta: Learning rate (e.g., 3.0).
//test_data: Optional vector of (input, digit) pairs for evaluation.
//Logic:
//For each epoch:
//Shuffle training data to ensure randomness.
//Split into mini-batches (e.g., 60,000 images with size 10 → 6,000 mini-batches).
//For each mini-batch, call update_mini_batch.
//If test_data provided, evaluate accuracy (correct predictions / total).
//SGD vs. Gradient Descent: Instead of computing the gradient over all 60,000 images (slow), it uses small mini-batches, making updates faster and approximating the true gradient.
//MNIST Context: Each epoch trains on all 60,000 images, but in random mini-batches, adjusting weights to better classify digits.
void Network::SGD(std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>& training_data,
    int epochs, int mini_batch_size, double eta,
    const std::vector<std::pair<Eigen::VectorXd, int>>* test_data) {
    size_t n = training_data.size();
    size_t n_test = test_data ? test_data->size() : 0;
    for (int j = 0; j < epochs; ++j) {
        std::shuffle(training_data.begin(), training_data.end(), rng);
        for (size_t k = 0; k < n; k += mini_batch_size) {
            std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> mini_batch(
                training_data.begin() + k,
                training_data.begin() + std::min(k + mini_batch_size, n));
            update_mini_batch(mini_batch, eta);
        }
        if (test_data) {
            std::cout << "Epoch " << j << ": (Correct Predictions) " << evaluate(*test_data) << " / " << n_test << std::endl;
        }
        else {
            std::cout << "Epoch " << j << " complete" << std::endl;
        }
    }
}

// Updates the network's weights and biases for a single mini-batch using gradient descent.
// mini_batch: A vector of (input, label) pairs, e.g., 10 MNIST images (784x1) and their one-hot labels (10x1).
// eta: Learning rate, controls the step size of updates (e.g., 3.0).
void Network::update_mini_batch(const std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>& mini_batch, double eta) {

    if (mini_batch.empty()) return;
    
    //Initializes gradient accumulators(weight_grads, bias_grads) with sizes matching each layer’s weight matrix and bias vector.
    std::vector<Eigen::MatrixXd> weight_grads;
    std::vector<Eigen::VectorXd> bias_grads;

    for (size_t i = 0; i < layers.size(); ++i) {
        weight_grads.emplace_back(Eigen::MatrixXd::Zero(layers[i].get_num_neurons(), layers[i].get_num_inputs()));
        bias_grads.emplace_back(Eigen::VectorXd::Zero(layers[i].get_num_neurons()));
    }

    // Accumulate gradients over the mini-batch
    for (const auto& [x, y] : mini_batch) {
        auto [delta_nabla_b, delta_nabla_w] = backprop(x, y);
        for (size_t i = 0; i < layers.size(); ++i) {
            bias_grads[i] += delta_nabla_b[i];
            weight_grads[i] += delta_nabla_w[i];
        }
    }

    // Update each layer's parameters
    double scale = eta / mini_batch.size();
    for (size_t i = 0; i < layers.size(); ++i) {
        layers[i].update_parameters(weight_grads[i] * scale, bias_grads[i] * scale);
    }
}


//Computes gradients of the cost function for one training example.
std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::MatrixXd>> Network::backprop(
    const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
    
    // Initialize gradients
    std::vector<Eigen::VectorXd> nabla_b(layers.size());
    std::vector<Eigen::MatrixXd> nabla_w(layers.size());

    for (size_t i = 0; i < layers.size(); ++i) {
        nabla_b[i] = Eigen::VectorXd::Zero(layers[i].get_activations().size());
        nabla_w[i] = Eigen::MatrixXd::Zero(layers[i].get_activations().size(), i == 0 ? x.size() : layers[i - 1].get_activations().size());
    }

    // Feedforward pass
    Eigen::VectorXd activation = x;
    std::vector<Eigen::VectorXd> activations = { x };
    std::vector<Eigen::VectorXd> zs;

    for (size_t i = 0; i < layers.size(); ++i) {
        activation = layers[i].forward(activation); // Compute forward pass
        zs.push_back(layers[i].get_pre_activations()); // Store pre-activation (z)
        activations.push_back(layers[i].get_activations()); // Store activation
    }

    // Backward pass
    Eigen::VectorXd delta = cost_derivative(activations.back(), y).cwiseProduct(sigmoid_prime(zs.back()));
    nabla_b.back() = delta;
    nabla_w.back() = delta * activations[activations.size() - 2].transpose();

    for (int l = 2; l < num_layers; ++l) {
        const Eigen::VectorXd& z = zs[zs.size() - l];
        Eigen::VectorXd sp = sigmoid_prime(z);
        delta = (layers[layers.size() - l + 1].get_weights().transpose() * delta).cwiseProduct(sp);
        nabla_b[nabla_b.size() - l] = delta;
        nabla_w[nabla_w.size() - l] = delta * activations[activations.size() - l - 1].transpose();
    }

    return { nabla_b, nabla_w };
}


//Counts correct predictions on test data.
//Logic:
//For each test example:
//Run feedforward to get output.
//Predict the digit with the highest output value(argmax).
//Compare with true label (0–9).
//Return number of correct predictions.
//MNIST Context: Accuracy like 9500/10000 means 95% correct digit predictions.
int Network::evaluate(const std::vector<std::pair<Eigen::VectorXd, int>>& test_data) {
    int correct = 0;
    for (const auto& [x, y] : test_data) {
        Eigen::VectorXd output = feedforward(x);
        int predicted = std::distance(output.data(), std::max_element(output.data(), output.data() + output.size()));
        if (predicted == y) ++correct;
    }
    return correct;
}

//Computes derivative of cost with respect to output(activations) of the output layer - dc/da;
Eigen::VectorXd Network::cost_derivative(const Eigen::VectorXd& output_activations, const Eigen::VectorXd& y) const{
    return output_activations - y;
}

void Network::display_biases() const {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== Biases ===" << std::endl;
    for (size_t i = 0; i < layers.size(); ++i) {
        std::string layer_name = (i == layers.size() - 1) ? "Output Layer" : "Hidden Layer " + std::to_string(i + 1);
        std::cout << layer_name << ":\n" << layers[i].print_parameters(false);
    }
}

void Network::display_weights() const {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== Weights ===" << std::endl;
    for (size_t i = 0; i < layers.size(); ++i) {
        std::string from_layer = (i == 0) ? "Input Layer" : "Hidden Layer " + std::to_string(i);
        std::string to_layer = (i == layers.size() - 1) ? "Output Layer" : "Hidden Layer " + std::to_string(i + 1);
        std::cout << "From " << from_layer << " to " << to_layer << ":\n" << layers[i].print_parameters(false);
    }
}

void Network::display_layer_biases(int max_elements) const {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== Layer Biases ===" << std::endl;
    for (size_t i = 0; i < layers.size(); ++i) {
        std::string layer_name = (i == layers.size() - 1) ? "Output Layer" : "Hidden Layer " + std::to_string(i + 1);
        std::cout << layer_name << " (" << layers[i].get_num_neurons() << " biases):" << std::endl;
        const Eigen::VectorXd& biases = layers[i].get_biases();
        int display_count = std::min(static_cast<int>(biases.size()), max_elements);
        std::cout << "  [";
        for (int j = 0; j < display_count; ++j) {
            std::cout << biases(j);
            if (j < display_count - 1) std::cout << ", ";
        }
        if (display_count < biases.size()) std::cout << ", ...";
        std::cout << "]" << std::endl;
        if (display_count < biases.size()) {
            std::cout << "  (Truncated, total " << biases.size() << " biases)" << std::endl;
        }
        std::cout << std::endl;
    }
}

void Network::display_layer_weights(int max_elements) const {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== Layer Weights ===" << std::endl;
    for (size_t i = 0; i < layers.size(); ++i) {
        std::string from_layer = (i == 0) ? "Input Layer" : "Hidden Layer " + std::to_string(i);
        std::string to_layer = (i == layers.size() - 1) ? "Output Layer" : "Hidden Layer " + std::to_string(i + 1);
        std::cout << "From " << from_layer << " to " << to_layer
            << " (" << layers[i].get_num_neurons() << "x" << layers[i].get_num_inputs() << " matrix):" << std::endl;
        const Eigen::MatrixXd& weights = layers[i].get_weights();
        int max_rows = std::min(static_cast<int>(weights.rows()), max_elements);
        int max_cols = std::min(static_cast<int>(weights.cols()), max_elements);
        for (int r = 0; r < max_rows; ++r) {
            std::cout << "  [";
            for (int c = 0; c < max_cols; ++c) {
                std::cout << weights(r, c);
                if (c < max_cols - 1) std::cout << ", ";
            }
            if (max_cols < weights.cols()) std::cout << ", ...";
            std::cout << "]" << std::endl;
        }
        if (max_rows < weights.rows() || max_cols < weights.cols()) {
            std::cout << "  (Truncated, full size: " << weights.rows() << "x" << weights.cols() << ")" << std::endl;
        }
        std::cout << std::endl;
    }
}

// Displays the gradients computed by backprop for a single training example.
// x: Input vector (e.g., 784x1 MNIST image).
// y: Desired output (e.g., 10x1 one-hot label for a digit).
// Prints bias gradients (e.g., 30x1 for hidden, 10x1 for output) and weight gradients
// (e.g., 30x784 for input to hidden, 10x30 for hidden to output), with truncation for readability.
void Network::display_backprop_gradients(const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
    // Compute gradients using backprop for the given example.
    auto [nabla_b, nabla_w] = backprop(x, y);

    // Set output format: fixed-point, 4 decimal places for readability.
    std::cout << std::fixed << std::setprecision(4);

    // Display bias gradients.
    std::cout << "=== Bias Gradients (from backprop) ===" << std::endl;
    for (size_t i = 0; i < nabla_b.size(); ++i) {
        // Label the layer (e.g., Hidden Layer 1 or Output Layer).
        std::string layer_name = (i == nabla_b.size() - 1) ? "Output Layer" : "Hidden Layer " + std::to_string(i + 1);
        // Print the number of bias gradients (e.g., 30 for hidden, 10 for output).
        std::cout << layer_name << " (" << nabla_b[i].size() << " bias gradients):" << std::endl;
        // Print each gradient value with its index.
        for (int j = 0; j < nabla_b[i].size(); ++j) {
            std::cout << "  db[" << j << "] = " << nabla_b[i](j);
            if (j < nabla_b[i].size() - 1) std::cout << ",";
            // Truncate after 10 elements to avoid overwhelming output for large vectors (e.g., 30).
            if (j == 9 && nabla_b[i].size() > 10) {
                std::cout << " ... (truncated, total " << nabla_b[i].size() << " bias gradients)";
                break;
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    // Display weight gradients.
    std::cout << "=== Weight Gradients (from backprop) ===" << std::endl;
    for (size_t i = 0; i < nabla_w.size(); ++i) {
        // Label the source and destination layers (e.g., Input to Hidden Layer 1).
        std::string from_layer = (i == 0) ? "Input Layer" : "Hidden Layer " + std::to_string(i);
        std::string to_layer = (i == nabla_w.size() - 1) ? "Output Layer" : "Hidden Layer " + std::to_string(i + 1);
        // Print the matrix dimensions (e.g., 30x784).
        std::cout << "From " << from_layer << " to " << to_layer
            << " (" << nabla_w[i].rows() << "x" << nabla_w[i].cols() << " gradient matrix):" << std::endl;
        // Limit to 5x5 subset for large matrices (e.g., 30x784).
        int max_rows = std::min(static_cast<int>(nabla_w[i].rows()), 5);
        int max_cols = std::min(static_cast<int>(nabla_w[i].cols()), 5);
        // Print each row of the subset.
        for (int r = 0; r < max_rows; ++r) {
            std::cout << "  [";
            // Print each column value.
            for (int c = 0; c < max_cols; ++c) {
                std::cout << nabla_w[i](r, c);
                if (c < max_cols - 1) std::cout << ", ";
            }
            if (max_cols < nabla_w[i].cols()) std::cout << ", ...";
            std::cout << "]" << std::endl;
        }
        // Indicate truncation if the full matrix is larger.
        if (max_rows < nabla_w[i].rows() || max_cols < nabla_w[i].cols()) {
            std::cout << "  (Truncated, full size: " << nabla_w[i].rows() << "x" << nabla_w[i].cols() << ")" << std::endl;
        }
        std::cout << std::endl;
    }
}

Eigen::VectorXd sigmoid(const Eigen::VectorXd& z) {
    return z.unaryExpr([](double x) { return 1.0 / (1.0 + std::exp(-x)); });
}

//used in backpropagation.
Eigen::VectorXd sigmoid_prime(const Eigen::VectorXd& z) {
    Eigen::VectorXd sz = sigmoid(z);
    return sz.cwiseProduct(Eigen::VectorXd::Ones(sz.size()) - sz);
}