#include "Layer.hpp"
#include"utils.h"
#include"GPUComputationContext.hpp";
#include"CPUComputationContext.hpp"
#include <sstream>
#include <iomanip>
#include <cmath>


/**
 * @brief Constructs a layer with specified input size and number of neurons.
 * Initializes weights and biases using Xavier initialization.
 * @param num_inputs Size of input vector
 * @param num_neurons Number of neurons in the layer
 * @param seed Random seed for initialization
 */
Layer::Layer(int num_inputs, int num_neurons, const Activation* activation, ComputationContext* context, unsigned int seed)
    : num_inputs_(num_inputs), num_neurons_(num_neurons),
    weights_(num_neurons, num_inputs), biases_(num_neurons),
    activations_(Eigen::VectorXd::Zero(num_neurons)),
    pre_activations_(num_neurons), input_(num_inputs),
    rng_(seed), has_valid_activations_(false), activation_(activation), 
    context_(context), d_weights_(nullptr), d_biases_(nullptr) {

    // Xavier initialization
    double stddev = std::sqrt(2.0 / (num_inputs + 1));
    std::normal_distribution<double> dist(0.0, stddev);

    // Initialize weights and biases (unchanged)
    for (int i = 0; i < num_neurons; ++i) {
        for (int j = 0; j < num_inputs; ++j) {
            weights_(i, j) = dist(rng_);
        }
        biases_(i) = dist(rng_);
    }

    // Allocate GPU memory and copy data if using GPU context
    context_->allocate_vector(&d_input_, num_inputs_);
    context_->allocate_vector(&d_pre_activations_, num_neurons_);
    context_->allocate_vector(&d_activations_, num_neurons_);

    context->allocate_weights(&d_weights_, num_neurons, num_inputs);
    context_->allocate_biases(&d_biases_, num_neurons);
    context_->copy_weights_to_device(d_weights_, weights_);
    context_->copy_biases_to_device(d_biases_, biases_);
}

/**
 * @brief Destructor to free GPU memory.
 */
Layer::~Layer() {
    context_->free_vector(d_input_);
    context_->free_vector(d_pre_activations_);
    context_->free_vector(d_activations_);
    context_->free_weights(d_weights_);
    context_->free_biases(d_biases_);
}


/**
 * @brief Computes the forward pass, producing activations for the input.
 * Caches input and pre-activations for gradient computation.
 * @param input Input vector (num_inputs x 1)
 * @return Output activations (num_neurons x 1)
 */
Eigen::VectorXd Layer::forward(const Eigen::VectorXd& input) {
    input_ = input;
    if (dynamic_cast<GPUComputationContext*>(context_)) {
        // GPU path
        context_->copy_to_device(d_input_, input);
        context_->computeLinearGPU(d_weights_, d_input_, d_biases_, d_pre_activations_, num_neurons_, num_inputs_);
        context_->applyActivationGPU(d_pre_activations_, d_activations_, num_neurons_, activation_);
        context_->copy_to_host(activations_, d_activations_, num_neurons_);
        context_->copy_to_host(pre_activations_, d_pre_activations_, num_neurons_); // For backprop compatibility
    }
    else {
        // CPU path
        pre_activations_ = context_->computeLinear(weights_, input, biases_);
        activations_ = context_->applyActivation(pre_activations_, activation_);
    }
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
    Eigen::VectorXd activation_derives = context_->computeActivationDerivative(activations_, pre_activations_, activation_);
    context_->computeGradients(deltas, activation_derives, input_, weight_grads, bias_grads);
}

/**
 * @brief Updates weights and biases using pre-computed gradients.
 * @param weight_grads Weight gradients (num_neurons x num_inputs)
 * @param bias_grads Bias gradients (num_neurons)
 */
void Layer::update_parameters(const Eigen::MatrixXd& weight_grads,
    const Eigen::VectorXd& bias_grads, double scale) {
    context_->updateParameters(weights_, biases_, weight_grads, bias_grads, scale);
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
    if (d_weights_) {
        context_->copy_weights_to_device(d_weights_, weights_);
    }
}

void Layer::set_biases(const Eigen::VectorXd& biases) {
    assert(biases.size() == biases_.size());
    biases_ = biases;
    if (d_biases_) {
        context_->copy_biases_to_device(d_biases_, biases_);
    }
}