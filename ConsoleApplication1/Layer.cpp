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
    rng_(seed), has_valid_activations_(false), d_pre_activations_(nullptr), activation_(activation),
    context_(context), d_weights_(nullptr), d_biases_(nullptr), 
    d_input_(nullptr), d_derivatives(nullptr), d_dy(nullptr), 
    d_weight_grads_(nullptr), d_bias_grads_(nullptr), 
    is_gpu_context_(dynamic_cast<GPUComputationContext*>(context_) != nullptr){

    if (dynamic_cast<GPUComputationContext*>(context_)) {
        contextGPU_ = dynamic_cast<GPUComputationContext*>(context_);
    }
    else {
        contextCPU_ = dynamic_cast<CPUComputationContext*>(context_);
    }

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

    if (is_gpu_context_) {
        // Allocate GPU memory and copy data if using GPU context
        context_->allocate_vector(&d_input_, num_inputs_);
        context_->allocate_vector(&d_pre_activations_, num_neurons_);
        context_->allocate_vector(&d_activations_, num_neurons_);
        context_->allocate_vector(&d_derivatives, num_neurons_);
        context_->allocate_vector(&d_dy, num_neurons_);
        context_->allocate_weights(&d_weights_, num_neurons, num_inputs);
        context_->allocate_biases(&d_biases_, num_neurons);
        context_->allocate_weights(&d_weight_grads_, num_neurons, num_inputs); // Allocate gradient storage
        context_->allocate_biases(&d_bias_grads_, num_neurons);              // Allocate gradient storage

        context_->copy_weights_to_device(d_weights_, weights_);
        context_->copy_biases_to_device(d_biases_, biases_);

        // Initialize dy with ones (to compute raw derivative, not scaled by deltas) 
        Eigen::VectorXd ones = Eigen::VectorXd::Ones(num_neurons_);
        context_->copy_to_device(d_dy, ones);

        // Initialize gradient storage to zero
        contextGPU_->set_to_zero(d_weight_grads_, num_neurons_ * num_inputs_);
        contextGPU_->set_to_zero(d_bias_grads_, num_neurons_);
        /*Eigen::MatrixXd zero_weights(num_neurons, num_inputs);
        Eigen::VectorXd zero_biases(num_neurons);
        zero_weights.setZero();
        zero_biases.setZero();
        context_->copy_weights_to_device(d_weight_grads_, zero_weights);
        context_->copy_biases_to_device(d_bias_grads_, zero_biases);*/
    }
    
}

/**
 * @brief Destructor to free GPU memory.
 */
Layer::~Layer() {

    if (is_gpu_context_) {
        contextGPU_->free_vector(d_input_);
        contextGPU_->free_vector(d_pre_activations_);
        contextGPU_->free_vector(d_activations_);
        contextGPU_->free_weights(d_weights_);
        contextGPU_->free_biases(d_biases_);
        contextGPU_->free_biases(d_derivatives);
        contextGPU_->free_biases(d_dy);
        contextGPU_->free_weights(d_weight_grads_);
        contextGPU_->free_biases(d_bias_grads_);
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
    if (is_gpu_context_) {
        // GPU path
        contextGPU_->copy_to_device(d_input_, input_);

        contextGPU_->computeLinearGPU(d_weights_, d_input_, d_biases_, d_pre_activations_, num_neurons_, num_inputs_);
        contextGPU_->copy_to_host(pre_activations_, d_pre_activations_, num_neurons_); // For backprop compatibility

        contextGPU_->applyActivationGPU(d_pre_activations_, d_activations_, num_neurons_, activation_);
        contextGPU_->copy_to_host(activations_, d_activations_, num_neurons_);
    }
    else {
        // CPU path
        pre_activations_ = contextCPU_->computeLinear(weights_, input, biases_);
        activations_ = contextCPU_->applyActivation(pre_activations_, activation_);
    }
    has_valid_activations_ = true;
    return activations_;
}

void Layer::compute_gradients_cpu(const Eigen::VectorXd& deltas,
    Eigen::MatrixXd& weight_grads,
    Eigen::VectorXd& bias_grads) const
{
    Eigen::VectorXd activation_derives = contextCPU_->computeActivationDerivative(activations_, pre_activations_, activation_);
    contextCPU_->computeGradients(deltas, activation_derives, input_, weight_grads, bias_grads);
}

void Layer::compute_gradients_gpu(const Eigen::VectorXd& deltas) {
    // Compute derivative on device
    contextGPU_->computeActivationDerivativeGPU(d_activations_, d_pre_activations_, d_dy, d_derivatives, num_neurons_, activation_);

    // If deltas is non-empty, copy to device; else assume it's already in bias_grads or similar
    if (!deltas.isZero()) {
        contextGPU_->copy_to_device(d_bias_grads_, deltas);  // Reuse bias_grads as temp delta
    }

    contextGPU_->computeGradientsGPU(deltas, d_derivatives, d_input_,
        d_weight_grads_, d_bias_grads_,
        num_neurons_, num_inputs_);
}

void Layer::compute_gradients(const Eigen::VectorXd& deltas,
    Eigen::MatrixXd& weight_grads,
    Eigen::VectorXd& bias_grads) const {

    if (is_gpu_context_) {
        const_cast<Layer*>(this)->compute_gradients_gpu(deltas);  // Const cast for mutating device state
        
        // Copy to host if needed
        context_->copy_to_host(weight_grads, d_weight_grads_, num_neurons_, num_inputs_);
        context_->copy_to_host(bias_grads, d_bias_grads_, num_neurons_);
    }
    else {
        compute_gradients_cpu(deltas, weight_grads, bias_grads);
    }
}
//void Layer::compute_gradients(const Eigen::VectorXd& deltas,
//    Eigen::MatrixXd& weight_grads,
//    Eigen::VectorXd& bias_grads) const
//{
//    if (dynamic_cast<GPUComputationContext*>(context_)) {
//        // GPU path
//        Eigen::VectorXd activation_derives = context_->computeActivationDerivativeGPU(d_activations_, d_pre_activations_, d_dy, d_derivatives, num_neurons_, activation_);
//        context_->computeGradientsGPU(
//            deltas, d_derivatives, d_input_, 
//            d_weight_grads_, d_bias_grads_, 
//            num_neurons_, num_inputs_);
//
//        //copy gradients to host
//        context_->copy_weights_to_host(weight_grads, d_weight_grads_, num_neurons_, num_inputs_);
//        context_->copy_biases_to_host(bias_grads, d_bias_grads_, num_neurons_);
//    }
//    else {
//        // CPU path
//        Eigen::VectorXd activation_derives = context_->computeActivationDerivative(activations_, pre_activations_, activation_);
//        context_->computeGradients(deltas, activation_derives, input_, weight_grads, bias_grads);
//    }
//}

//void Layer::update_parameters(
//    const Eigen::MatrixXd& weight_grads,
//    const Eigen::VectorXd& bias_grads, 
//    double* accumulate_weight_grads,
//    double* accumulate_bias_grads, 
//    double scale) {
//
//    if (dynamic_cast<GPUComputationContext*>(context_)) {
//        if (!accumulate_weight_grads || !accumulate_bias_grads) {
//            //TODO: throw error
//        }
//        context_->updateParametersGPU(
//            d_weights_, d_biases_,
//            accumulate_weight_grads, accumulate_bias_grads,
//            num_neurons_, num_inputs_, num_neurons_, scale
//        );
//
//        // Update host weights and biases for compatibility
//        context_->copy_weights_to_host(weights_, d_weights_, num_neurons_, num_inputs_);
//        context_->copy_biases_to_host(biases_, d_biases_, num_neurons_);
//    }
//    else {
//        context_->updateParameters(weights_, biases_, weight_grads, bias_grads, scale);
//    }
//}

//CPU variant
void Layer::update_parameters(
    const Eigen::MatrixXd& weight_grads,
    const Eigen::VectorXd& bias_grads,
    double scale) {
    context_->updateParameters(weights_, biases_, weight_grads, bias_grads, scale);
}

//GPU variant
void Layer::update_parameters(
    double* accumulate_weight_grads,
    double* accumulate_bias_grads,
    double scale) {

    context_->updateParametersGPU(
        d_weights_, d_biases_,
        accumulate_weight_grads, accumulate_bias_grads,
        num_neurons_, num_inputs_, num_neurons_, scale
    );

    // Update host weights and biases for compatibility
    context_->copy_weights_to_host(weights_, d_weights_, num_neurons_, num_inputs_);
    context_->copy_biases_to_host(biases_, d_biases_, num_neurons_);
}

// New: Apply derivative elementwise on GPU (for backprop delta propagation)
void Layer::apply_derivative_gpu(double* d_delta) {
    contextGPU_->computeActivationDerivativeGPU(d_activations_, d_pre_activations_, d_dy, d_derivatives, num_neurons_, activation_);
    // Elementwise multiply d_delta *= d_derivatives
    contextGPU_->launch_elementwise_multiply(d_delta, d_derivatives, d_delta, num_neurons_);
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

void Layer::set_pre_activations(const Eigen::VectorXd& pre_activations) {
    assert(pre_activations.size() == pre_activations_.size());
    pre_activations_ = pre_activations;
    if (d_pre_activations_) {
        context_->copy_to_device(d_pre_activations_, pre_activations);
    }
}

void Layer::set_activations(const Eigen::VectorXd& activations) {
    assert(activations.size() == activations_.size());
    activations_ = activations;
    if (d_activations_) {
        context_->copy_to_device(d_activations_, activations);
    }
}