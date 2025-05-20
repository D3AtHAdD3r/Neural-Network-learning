#include "SigmoidNeuron.hpp"
#include <cmath>
#include <sstream>
#include <iomanip>

SigmoidNeuron::SigmoidNeuron(int num_inputs, unsigned int seed)
	: weights_(num_inputs), bias_(0.0), activation_(0.0), rng_(seed) {
	// Initialize weights and bias with random values in [-1, 1]
	std::uniform_real_distribution<double> dist(-1.0, 1.0);
	for (int i = 0; i < num_inputs; ++i) {
		weights_(i) = dist(rng_);
	}
	bias_ = dist(rng_);
	// TODO: Consider Xavier initialization for better convergence
}


double SigmoidNeuron::sigmoid(double x) const {
	return 1.0 / (1.0 + std::exp(-x));
}


double SigmoidNeuron::forward(const Eigen::VectorXd& input) {
	// Compute weighted sum: w * x + b
	double z = weights_.dot(input) + bias_;

	// Apply sigmoid activation
	activation_ = sigmoid(z);
	return activation_;
}


void SigmoidNeuron::compute_gradient(double delta, const Eigen::VectorXd& input, Eigen::VectorXd& weight_grad, double& bias_grad) const {

	// Gradient of sigmoid: sigmoid(x) * (1 - sigmoid(x))
	double sigmoid_deriv = activation_ * (1.0 - activation_);

	// Delta scaled by sigmoid derivative
	double adjusted_delta = delta * sigmoid_deriv;

	// Gradient for weights: delta * sigmoid'(z) * input
	weight_grad = adjusted_delta * input;

	// Gradient for bias: delta * sigmoid'(z)
	bias_grad = adjusted_delta;
}

void SigmoidNeuron::update_parameters(const Eigen::VectorXd& weights, double bias)
{
	// Set pre-computed weights and bias
	weights_ = weights;
	bias_ = bias;
}


std::string SigmoidNeuron::print_parameters(bool json_format) const {
	std::stringstream ss;
	if (json_format) {
		// JSON-like output
		ss << "{\n";
		ss << "  \"weights\": [";
		for (int i = 0; i < weights_.size(); ++i) {
			ss << weights_(i);
			if (i < weights_.size() - 1) ss << ", ";
		}
		ss << "],\n";
		ss << "  \"bias\": " << bias_ << ",\n";
		ss << "  \"activation\": " << activation_ << "\n}";
	}
	else {
		// Text output similar to original Network.cpp
		ss << "Weights: ";
		for (int i = 0; i < weights_.size(); ++i) {
			ss << std::fixed << std::setprecision(4) << weights_(i) << " ";
		}
		ss << "\nBias: " << std::fixed << std::setprecision(4) << bias_ << "\n";
		ss << "Activation: " << std::fixed << std::setprecision(4) << activation_ << "\n";
	}
	return ss.str();
}