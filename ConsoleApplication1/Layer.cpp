#include "Layer.hpp"
#include <sstream>
#include <iomanip>
#include <cmath>

/**
 * @brief Sigmoid activation function.
 * @param x Input value
 * @return Sigmoid of x (1 / (1 + exp(-x)))
 */
double Layer::sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

/**
 * @brief Constructs a layer with specified input size and number of neurons.
 * Initializes weights and biases using Xavier initialization.
 * @param num_inputs Size of input vector
 * @param num_neurons Number of neurons in the layer
 * @param seed Random seed for initialization
 */
Layer::Layer(int num_inputs, int num_neurons, unsigned int seed)
    : num_inputs_(num_inputs), num_neurons_(num_neurons),
    weights_(num_neurons, num_inputs), biases_(num_neurons),
    activations_(Eigen::VectorXd::Zero(num_neurons)),
    pre_activations_(num_neurons), input_(num_inputs),
    rng_(seed), has_valid_activations_(false) {
    // Xavier initialization
    double stddev = std::sqrt(2.0 / (num_inputs + 1));
    std::normal_distribution<double> dist(0.0, stddev);

    // Initialize weights
    for (int i = 0; i < num_neurons; ++i) {
        for (int j = 0; j < num_inputs; ++j) {
            weights_(i, j) = dist(rng_);
        }
    }

    // Initialize biases
    for (int i = 0; i < num_neurons; ++i) {
        biases_(i) = dist(rng_);
    }
}

/**
 * @brief Computes the forward pass, producing activations for the input.
 * Caches input and pre-activations for gradient computation.
 * @param input Input vector (num_inputs x 1)
 * @return Output activations (num_neurons x 1)
 */
Eigen::VectorXd Layer::forward(const Eigen::VectorXd& input) {
    input_ = input;
    pre_activations_ = weights_ * input + biases_;
    activations_ = pre_activations_.unaryExpr([](double x) { return sigmoid(x); });
    has_valid_activations_ = true;
    return activations_;
}

/**
 * @brief Computes gradients for weights and biases based on backpropagated errors.
 * @param deltas Error terms from the next layer or cost function
 * @param weight_grads Output weight gradients (num_neurons x num_inputs)
 * @param bias_grads Output bias gradients (num_neurons)
 */
void Layer::compute_gradients(const Eigen::VectorXd& deltas,
    Eigen::MatrixXd& weight_grads,
    Eigen::VectorXd& bias_grads) const {
    Eigen::VectorXd sigmoid_derivs = activations_.cwiseProduct(Eigen::VectorXd::Ones(num_neurons_) - activations_);
    Eigen::VectorXd adjusted_deltas = deltas.cwiseProduct(sigmoid_derivs);
    weight_grads = adjusted_deltas * input_.transpose();
    bias_grads = adjusted_deltas;
}

/**
 * @brief Updates weights and biases using pre-computed gradients.
 * @param weight_grads Weight gradients (num_neurons x num_inputs)
 * @param bias_grads Bias gradients (num_neurons)
 */
void Layer::update_parameters(const Eigen::MatrixXd& weight_grads,
    const Eigen::VectorXd& bias_grads) {
    weights_ -= weight_grads;
    biases_ -= bias_grads;
}

/**
 * @brief Prints layer parameters (weights, biases, activations).
 * If activations are not computed, indicates "not computed".
 * @param json_format If true, output in JSON-like format; else, text format
 * @return String representation of parameters
 */
std::string Layer::print_parameters(bool json_format) const {
    std::stringstream ss;
    if (json_format) {
        ss << "{\n  \"neurons\": [\n";
        for (int i = 0; i < num_neurons_; ++i) {
            ss << "    {\n";
            ss << "      \"weights\": [";
            for (int j = 0; j < num_inputs_; ++j) {
                ss << weights_(i, j);
                if (j < num_inputs_ - 1) ss << ", ";
            }
            ss << "],\n";
            ss << "      \"bias\": " << biases_(i) << ",\n";
            ss << "      \"activation\": " << (has_valid_activations_ ? std::to_string(activations_(i)) : "\"not computed\"") << "\n";
            ss << "    }";
            if (i < num_neurons_ - 1) ss << ",";
            ss << "\n";
        }
        ss << "  ]\n}";
    }
    else {
        ss << "Layer (" << num_neurons_ << " neurons, " << num_inputs_ << " inputs):\n";
        for (int i = 0; i < num_neurons_; ++i) {
            ss << "Neuron " << i << ":\n";
            ss << "Weights: ";
            for (int j = 0; j < num_inputs_; ++j) {
                ss << std::fixed << std::setprecision(4) << weights_(i, j) << " ";
            }
            ss << "\nBias: " << std::fixed << std::setprecision(4) << biases_(i) << "\n";
            ss << "Activation: ";
            if (has_valid_activations_) {
                ss << std::fixed << std::setprecision(4) << activations_(i);
            }
            else {
                ss << "not computed";
            }
            ss << "\n";
        }
    }
    return ss.str();
}

void Layer::set_weights(const Eigen::MatrixXd& weights)
{
    assert(weights.rows() == weights_.rows() && weights.cols() == weights_.cols());
    weights_ = weights;
}

void Layer::set_biases(const Eigen::VectorXd& biases) {
    assert(biases.size() == biases_.size());
    biases_ = biases;
}