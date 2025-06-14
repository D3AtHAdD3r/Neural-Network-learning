#include "Network.hpp"
#include <iomanip>

/**
 * @brief Constructs a network with specified layer sizes.
 * Initializes layers with Xavier-initialized weights and biases.
 * @param sizes Vector of layer sizes (e.g., {784, 30, 10} for MNIST)
 */
Network::Network(const std::vector<int>& sizes, double lambda) : sizes(sizes), num_layers(sizes.size()), rng(std::random_device{}()), last_test_loss(0.0), lambda(lambda) {
    for (size_t i = 1; i < sizes.size(); ++i) {
        layers.emplace_back(sizes[i - 1], sizes[i], static_cast<unsigned int>(rng()));
    }
}

/**
 * @brief Computes the network output for a given input.
 * Passes the input through each layer's forward pass.
 * @param a Input vector
 * @return Output activations of the final layer
 */
Eigen::VectorXd Network::feedforward(const Eigen::VectorXd& a) {
    Eigen::VectorXd activation = a;
    for (auto& layer : layers) {
        activation = layer.forward(activation);
    }
    return activation;
}

/**
 * @brief Trains the network using stochastic gradient descent.
 * Shuffles training data and updates parameters via mini-batches.
 * @param training_data Vector of (input, target) pairs
 * @param epochs Number of training epochs
 * @param mini_batch_size Size of each mini-batch
 * @param eta Learning rate
 * @param test_data Optional test data for evaluation
 * @param verbose If true, display detailed metrics per epoch
 */
void Network::SGD(std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>& training_data,
    int epochs, int mini_batch_size, double eta,
    const std::vector<std::pair<Eigen::VectorXd, int>>* test_data, 
    bool verbose) {
    size_t n = training_data.size();
    size_t n_test = test_data ? test_data->size() : 0;
    for (int j = 0; j < epochs; ++j) {
        std::shuffle(training_data.begin(), training_data.end(), rng);
        double batch_gradient_norm = 0.0;

        size_t num_batches = (n + mini_batch_size - 1) / mini_batch_size; // Ceiling division

        for (size_t k = 0; k < n; k += mini_batch_size) {
            std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> mini_batch(
                training_data.begin() + k,
                training_data.begin() + std::min(k + mini_batch_size, n));

            batch_gradient_norm += update_mini_batch(mini_batch, eta, n);
        }

        batch_gradient_norm /= num_batches;
       
        if (verbose && test_data) {
            auto [correct, total_loss] = evaluate(*test_data, n);
            double accuracy = (n_test > 0) ? (correct * 100.0 / n_test) : 0.0;
            double loss = (n_test > 0) ? total_loss / n_test : 0.0;
            std::cout << std::fixed << std::setprecision(4);
            std::cout << "Epoch " << j
                << ": Accuracy = " << accuracy << "%"
                << ", Correct = " << correct << "/" << n_test
                << ", Loss = " << loss
                << ", Gradient Norm = " << batch_gradient_norm;
            if (lambda > 0.0) {
                std::cout << ", Lambda = " << lambda;
            }
            std::cout << std::endl;
        }
        else if (test_data) {
            auto [correct, total_loss] = evaluate(*test_data, n);
            std::cout << "Epoch " << j << ": Correct Predictions = " << correct << "/" << n_test << std::endl;
        }
        else {
            std::cout << "Epoch " << j << " complete" << std::endl;
        }
    }
}

/**
 * @brief Updates weights and biases for a mini-batch and computes gradient norm.
 * @param mini_batch Vector of (input, target) pairs
 * @param eta Learning rate
 * @param n Number of training examples for L2 scaling
 * @return L2 norm of gradients for the mini-batch
 */
double Network::update_mini_batch(const std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>& mini_batch, double eta, size_t n) {
    if (mini_batch.empty()) return 0.0;

    std::vector<Eigen::MatrixXd> weight_grads;
    std::vector<Eigen::VectorXd> bias_grads;
    for (size_t i = 0; i < layers.size(); ++i) {
        weight_grads.emplace_back(Eigen::MatrixXd::Zero(layers[i].get_num_neurons(), layers[i].get_num_inputs()));
        bias_grads.emplace_back(Eigen::VectorXd::Zero(layers[i].get_num_neurons()));
    }

    for (const auto& [x, y] : mini_batch) {
        auto [delta_nabla_b, delta_nabla_w] = backprop(x, y, n);
        for (size_t i = 0; i < layers.size(); ++i) {
            bias_grads[i] += delta_nabla_b[i];
            weight_grads[i] += delta_nabla_w[i];
        }
    }

    double norm = 0.0;
    for (size_t i = 0; i < layers.size(); ++i) {
        norm += bias_grads[i].squaredNorm();
        norm += weight_grads[i].squaredNorm();
    }
    norm = std::sqrt(norm / mini_batch.size());

    double scale = eta / mini_batch.size();
    for (size_t i = 0; i < layers.size(); ++i) {
        layers[i].update_parameters(weight_grads[i] * scale, bias_grads[i] * scale);
    }

    return norm;
}

/**
 * @brief Computes gradients for a single training example using backpropagation.
 * @param x Input vector
 * @param y Target vector
 * @param n Number of training examples for L2 scaling
 * @return Pair of bias gradients and weight gradients for each layer
 */
std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::MatrixXd>> Network::backprop(
    const Eigen::VectorXd& x, const Eigen::VectorXd& y, size_t n) {
    std::vector<Eigen::VectorXd> nabla_b(layers.size());
    std::vector<Eigen::MatrixXd> nabla_w(layers.size());

    for (size_t i = 0; i < layers.size(); ++i) {
        nabla_b[i] = Eigen::VectorXd::Zero(layers[i].get_activations().size());
        nabla_w[i] = Eigen::MatrixXd::Zero(layers[i].get_activations().size(), i == 0 ? x.size() : layers[i - 1].get_activations().size());
    }

    Eigen::VectorXd activation = x;
    std::vector<Eigen::VectorXd> activations = { x };
    std::vector<Eigen::VectorXd> zs;

    for (size_t i = 0; i < layers.size(); ++i) {
        activation = layers[i].forward(activation);
        zs.push_back(layers[i].get_pre_activations());
        activations.push_back(layers[i].get_activations());
    }

    Eigen::VectorXd delta = cost_derivative(activations.back(), y).cwiseProduct(sigmoid_prime(zs.back()));
    nabla_b.back() = delta;
    nabla_w.back() = delta * activations[activations.size() - 2].transpose();
    if (lambda > 0.0 && n > 0) {
        nabla_w.back() += (lambda / n) * layers[layers.size() - 1].get_weights(); // Scaled L2
    }
    

    for (int l = 2; l < num_layers; ++l) {
        const Eigen::VectorXd& z = zs[zs.size() - l];
        Eigen::VectorXd sp = sigmoid_prime(z);
        delta = (layers[layers.size() - l + 1].get_weights().transpose() * delta).cwiseProduct(sp);
        nabla_b[nabla_b.size() - l] = delta;
        nabla_w[nabla_w.size() - l] = delta * activations[activations.size() - l - 1].transpose();
        if (lambda > 0.0 && n > 0) {
            nabla_w[nabla_w.size() - l] += (lambda / n) * layers[layers.size() - l].get_weights(); // Scaled L2
        }
    }

    return { nabla_b, nabla_w };
}

/**
 * @brief Evaluates the network on test data and computes loss.
 * @param test_data Vector of (input, label) pairs
 * @param n Number of training examples for L2 regularization scaling
 * @return Pair of (correct predictions, total MSE loss including regularization)
 */
std::pair<int, double> Network::evaluate(const std::vector<std::pair<Eigen::VectorXd, int>>& test_data, size_t n) {
    int correct = 0;
    double total_loss = 0.0;

    double weight_norm = 0.0;
    for (const auto& layer : layers) {
        weight_norm += layer.get_weights().squaredNorm();
    }


    for (const auto& [x, y] : test_data) {
        Eigen::VectorXd output = feedforward(x);
        int predicted = std::distance(output.data(), std::max_element(output.data(), output.data() + output.size()));
        if (predicted == y) ++correct;

        Eigen::VectorXd target = Eigen::VectorXd::Zero(output.size());
        target(y) = 1.0; // One-hot encoding for target label
        Eigen::VectorXd diff = output - target;
        total_loss += diff.squaredNorm();
    }
    if (lambda > 0.0 && n > 0) {
        total_loss += 0.5 * lambda * weight_norm / n; // Scaled L2 regularization
    }
    
    last_test_loss = total_loss; // Cache total loss
    return { correct, total_loss };
}

/**
 * @brief Computes the derivative of the cost function w.r.t. output activations.
 * @param output_activations Output activations of the final layer
 * @param y Target vector
 * @return Cost derivative
 */
Eigen::VectorXd Network::cost_derivative(const Eigen::VectorXd& output_activations, const Eigen::VectorXd& y) const {
    return output_activations - y;
}

/**
 * @brief Computes the mean squared error loss over test data.
 * @param test_data Vector of (input, label) pairs
 * @return Average MSE loss
 */
double Network::compute_test_loss(const std::vector<std::pair<Eigen::VectorXd, int>>& test_data)
{
    double total_loss = 0.0;
    for (const auto& [x, y] : test_data) {
        Eigen::VectorXd output = feedforward(x);
        Eigen::VectorXd target = Eigen::VectorXd::Zero(output.size());
        target(y) = 1.0; // One-hot encoding for target label
        Eigen::VectorXd diff = output - target;
        total_loss += diff.squaredNorm();
    }
    return total_loss / test_data.size();

}


/**
 * @brief Computes the L2 norm of gradients for a mini-batch.
 * @param mini_batch Vector of (input, target) pairs
 * @param n Number of training examples for L2 regularization scaling
 * @return L2 norm of gradients
 */
double Network::compute_gradient_norm(const std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>& mini_batch, size_t n)
{
    std::vector<Eigen::MatrixXd> weight_grads;
    std::vector<Eigen::VectorXd> bias_grads;
    for (size_t i = 0; i < layers.size(); ++i) {
        weight_grads.emplace_back(Eigen::MatrixXd::Zero(layers[i].get_num_neurons(), layers[i].get_num_inputs()));
        bias_grads.emplace_back(Eigen::VectorXd::Zero(layers[i].get_num_neurons()));
    }

    for (const auto& [x, y] : mini_batch) {
        auto [delta_nabla_b, delta_nabla_w] = backprop(x, y, n);
        for (size_t i = 0; i < layers.size(); ++i) {
            bias_grads[i] += delta_nabla_b[i];
            weight_grads[i] += delta_nabla_w[i];
        }
    }

    double norm = 0.0;
    for (size_t i = 0; i < layers.size(); ++i) {
        norm += bias_grads[i].squaredNorm();
        norm += weight_grads[i].squaredNorm();
    }
    return std::sqrt(norm / mini_batch.size());
}

/**
 * @brief Displays biases for all layers.
 */
void Network::display_biases() const {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== Biases ===" << std::endl;
    for (size_t i = 0; i < layers.size(); ++i) {
        std::string layer_name = (i == layers.size() - 1) ? "Output Layer" : "Hidden Layer " + std::to_string(i + 1);
        std::cout << layer_name << ":\n" << layers[i].print_parameters(false);
    }
}

/**
 * @brief Displays weights for all layers.
 */
void Network::display_weights() const {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== Weights ===" << std::endl;
    for (size_t i = 0; i < layers.size(); ++i) {
        std::string from_layer = (i == 0) ? "Input Layer" : "Hidden Layer " + std::to_string(i);
        std::string to_layer = (i == layers.size() - 1) ? "Output Layer" : "Hidden Layer " + std::to_string(i + 1);
        std::cout << "From " << from_layer << " to " << to_layer << ":\n" << layers[i].print_parameters(false);
    }
}

/**
 * @brief Displays biases layer-wise with truncation.
 * @param max_elements Maximum elements to display per layer
 */
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

/**
 * @brief Displays weights layer-wise with truncation.
 * @param max_elements Maximum elements to display per layer
 */
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

/**
 * @brief Displays gradients computed by backpropagation for a single example.
 * @param x Input vector
 * @param y Target vector
 * @param n Number of training examples for L2 regularization scaling
 */
void Network::display_backprop_gradients(const Eigen::VectorXd& x, const Eigen::VectorXd& y, size_t n) {
    auto [nabla_b, nabla_w] = backprop(x, y, n);
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== Bias Gradients (from backprop) ===" << std::endl;
    for (size_t i = 0; i < nabla_b.size(); ++i) {
        std::string layer_name = (i == nabla_b.size() - 1) ? "Output Layer" : "Hidden Layer " + std::to_string(i + 1);
        std::cout << layer_name << " (" << nabla_b[i].size() << " bias gradients):" << std::endl;
        for (int j = 0; j < nabla_b[i].size(); ++j) {
            std::cout << "  db[" << j << "] = " << nabla_b[i](j);
            if (j < nabla_b[i].size() - 1) std::cout << ",";
            if (j == 9 && nabla_b[i].size() > 10) {
                std::cout << " ... (truncated, total " << nabla_b[i].size() << " bias gradients)";
                break;
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    std::cout << "=== Weight Gradients (from backprop) ===" << std::endl;
    for (size_t i = 0; i < nabla_w.size(); ++i) {
        std::string from_layer = (i == 0) ? "Input Layer" : "Hidden Layer " + std::to_string(i);
        std::string to_layer = (i == nabla_w.size() - 1) ? "Output Layer" : "Hidden Layer " + std::to_string(i + 1);
        std::cout << "From " << from_layer << " to " << to_layer
            << " (" << nabla_w[i].rows() << "x" << nabla_w[i].cols() << " gradient matrix):" << std::endl;
        int max_rows = std::min(static_cast<int>(nabla_w[i].rows()), 5);
        int max_cols = std::min(static_cast<int>(nabla_w[i].cols()), 5);
        for (int r = 0; r < max_rows; ++r) {
            std::cout << "  [";
            for (int c = 0; c < max_cols; ++c) {
                std::cout << nabla_w[i](r, c);
                if (c < max_cols - 1) std::cout << ", ";
            }
            if (max_cols < nabla_w[i].cols()) std::cout << ", ...";
            std::cout << "]" << std::endl;
        }
        if (max_rows < nabla_w[i].rows() || max_cols < nabla_w[i].cols()) {
            std::cout << "  (Truncated, full size: " << nabla_w[i].rows() << "x" << nabla_w[i].cols() << ")" << std::endl;
        }
        std::cout << std::endl;
    }
}

/**
 * @brief Applies sigmoid activation element-wise to a vector.
 * @param z Input vector
 * @return Sigmoid of each element
 */
Eigen::VectorXd sigmoid(const Eigen::VectorXd& z) {
    return z.unaryExpr([](double x) { return 1.0 / (1.0 + std::exp(-x)); });
}

/**
 * @brief Computes the derivative of the sigmoid function element-wise.
 * @param z Input vector
 * @return Sigmoid derivative for each element
 */
Eigen::VectorXd sigmoid_prime(const Eigen::VectorXd& z) {
    Eigen::VectorXd sz = sigmoid(z);
    return sz.cwiseProduct(Eigen::VectorXd::Ones(sz.size()) - sz);
}