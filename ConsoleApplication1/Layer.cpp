#include "Layer.hpp"
#include <sstream>
#include <iomanip>

Layer::Layer(int num_inputs, int num_neurons, unsigned int seed)
	: num_inputs_(num_inputs), num_neurons_(num_neurons),
	weights_(num_neurons, num_inputs), biases_(num_neurons),
	activations_(num_neurons) {

	// Initialize neurons and populate weight/bias matrices
	for (int i = 0; i < num_neurons; ++i) {
		neurons_.emplace_back(num_inputs, seed + i); // Unique seed per neuron
		weights_.row(i) = neurons_[i].get_weights();; // Copy weights
		biases_(i) = neurons_[i].get_bias(); // Copy bias
	}
}



Eigen::VectorXd Layer::forward(const Eigen::VectorXd& input) {

	// Batch forward pass: compute activations for all neurons
	// z = W * x + b
	Eigen::VectorXd z = weights_ * input + biases_;
	// Apply sigmoid to each element
	for (int i = 0; i < num_neurons_; ++i) {
		activations_(i) = 1.0 / (1.0 + std::exp(-z(i)));
		// Update neuron’s cached activation for consistency
		neurons_[i].forward(input); // Ensures neuron state is updated
	}
	return activations_;
}

void Layer::compute_gradients(const Eigen::VectorXd& deltas,
	Eigen::MatrixXd& weight_grads,
	Eigen::VectorXd& bias_grads) const {

	// Batch gradient computation
	weight_grads.resize(num_neurons_, num_inputs_);
	bias_grads.resize(num_neurons_);

	for (int i = 0; i < num_neurons_; ++i) {
		Eigen::VectorXd w_grad;
		double b_grad;
		neurons_[i].compute_gradient(deltas(i), w_grad, b_grad);
		weight_grads.row(i) = w_grad;
		bias_grads(i) = b_grad;
	}
}


void Layer::update_parameters(const Eigen::MatrixXd& weight_grads,
	const Eigen::VectorXd& bias_grads,
	double learning_rate) {

	// Update weights and biases for all neurons
	for (int i = 0; i < num_neurons_; ++i) {
		neurons_[i].update_parameters(weight_grads.row(i), bias_grads(i), learning_rate);
		// Sync cached matrices with updated neuron parameters
		weights_.row(i) = neurons_[i].get_weights();
		biases_(i) = neurons_[i].get_bias();
	}
}

std::string Layer::print_parameters(bool json_format) const {
	std::stringstream ss;
	if (json_format) {
		// JSON-like output
		ss << "{\n  \"neurons\": [\n";
		for (size_t i = 0; i < neurons_.size(); ++i) {
			ss << "    " << neurons_[i].print_parameters(true);
			if (i < neurons_.size() - 1) ss << ",";
			ss << "\n";
		}
		ss << "  ]\n}";
	}
	else {
		// Text output similar to Network.cpp
		ss << "Layer (" << num_neurons_ << " neurons, " << num_inputs_ << " inputs):\n";
		for (size_t i = 0; i < neurons_.size(); ++i) {
			ss << "Neuron " << i << ":\n" << neurons_[i].print_parameters(false);
		}
	}
	return ss.str();
}