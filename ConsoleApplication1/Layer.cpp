#include "Layer.hpp"
#include <sstream>
#include <iomanip>

double Layer::sigmoid(double x) {
	return 1.0 / (1.0 + std::exp(-x));
}

Layer::Layer(int num_inputs, int num_neurons, unsigned int seed)
	: num_inputs_(num_inputs), num_neurons_(num_neurons),
	weights_(num_neurons, num_inputs), biases_(num_neurons),
	activations_(num_neurons), pre_activations_(num_neurons), input_(num_inputs) {

	// Initialize neurons and populate weight/bias matrices
	for (int i = 0; i < num_neurons; ++i) {
		neurons_.emplace_back(num_inputs, seed + i); // Unique seed per neuron
		weights_.row(i) = neurons_[i].get_weights();; // Copy weights
		biases_(i) = neurons_[i].get_bias(); // Copy bias
	}
}


Eigen::VectorXd Layer::forward(const Eigen::VectorXd& input) {

	// Cache input in neurons for gradient computation
	input_ = input;

	// Batch forward pass: z = W * x + b
	pre_activations_ = weights_ * input + biases_;

	// Apply sigmoid element-wise
	activations_ = pre_activations_.unaryExpr([](double x) { return sigmoid(x); });

	// Update neuron activations for state consistency
	for (int i = 0; i < num_neurons_; ++i) {
		neurons_[i].set_activation(activations_(i));
	}
	return activations_;
}


void Layer::compute_gradients(const Eigen::VectorXd& deltas,
	Eigen::MatrixXd& weight_grads,
	Eigen::VectorXd& bias_grads) const {

	// Compute sigmoid derivatives: sigma'(z) = a * (1 - a)
	Eigen::VectorXd sigmoid_derivs = activations_.cwiseProduct(Eigen::VectorXd::Ones(num_neurons_) - activations_);

	// Adjusted deltas: delta * sigma'(z)
	Eigen::VectorXd adjusted_deltas = deltas.cwiseProduct(sigmoid_derivs);

	// Weight gradients: delta * sigma'(z) * input^T
	weight_grads = adjusted_deltas * input_.transpose();

	// Bias gradients: delta * sigma'(z)
	bias_grads = adjusted_deltas;
}


void Layer::update_parameters(const Eigen::MatrixXd& weight_grads,
	const Eigen::VectorXd& bias_grads) {
	// Update weights and biases using gradient descent
	weights_ -= weight_grads;
	biases_ -= bias_grads;

	// Sync neuron parameters with pre-computed values
	for (int i = 0; i < num_neurons_; ++i) {
		neurons_[i].update_parameters(weights_.row(i), biases_(i));
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