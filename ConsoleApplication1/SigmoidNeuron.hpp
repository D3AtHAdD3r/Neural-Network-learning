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
	void compute_gradient(double delta, const Eigen::VectorXd& input, Eigen::VectorXd& weight_grad, double& bias_grad) const;

	// Update weights and bias with pre-computed values
	void update_parameters(const Eigen::VectorXd& weights, double bias);

	// Print parameters (weights, bias, activation) in text or JSON format
	std::string print_parameters(bool json_format = false) const;

	// Getters
	double get_activation() const { return activation_; };
	const Eigen::VectorXd& get_weights() const { return weights_; };
	double get_bias() const { return bias_; };

	// Setters for state synchronization
	void set_activation(double activation) { activation_ = activation; };
};

#endif // SIGMOID_NEURON_HPP