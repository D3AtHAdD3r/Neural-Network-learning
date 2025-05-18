#pragma once
#ifndef SIGMOID_NEURON_HPP
#define SIGMOID_NEURON_HPP

#include <Eigen/Dense>
#include <random>
#include <string>

class SigmoidNeuron
{
private:
	Eigen::VectorXd weights_; // Weight vector for inputs
	double bias_;             // Bias term
	Eigen::VectorXd input_;   // Cached input for gradient computation
	double activation_;       // Cached output after sigmoid
	std::mt19937 rng_;        // Random number generator for initialization

	// Sigmoid activation function
	double sigmoid(double x) const;

public:
	// Constructor: num_inputs is the size of input vector
	SigmoidNeuron(int num_inputs, unsigned int seed = 42);

	// Forward pass: compute activation for given input
	double forward(const Eigen::VectorXd& input);

	// Compute gradients for weights and bias
	void compute_gradient(double delta, Eigen::VectorXd& weight_grad, double& bias_grad) const;

	// Update weights and bias using gradients and learning rate
	void update_parameters(const Eigen::VectorXd& weight_grad, double bias_grad, double learning_rate);

	// Print parameters (weights, bias, activation) in text or JSON format
	std::string print_parameters(bool json_format = false) const;

	// Get activation (for Layer access)
	double get_activation() const { return activation_; }

	// Getters for weights and bias (for Layer initialization)
	const Eigen::VectorXd& get_weights() const { return weights_; }
	double get_bias() const { return bias_; }
};

#endif // SIGMOID_NEURON_HPP