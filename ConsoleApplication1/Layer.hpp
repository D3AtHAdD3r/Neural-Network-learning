#pragma once
#ifndef LAYER_HPP
#define LAYER_HPP

#include "SigmoidNeuron.hpp"
#include <vector>
#include <Eigen/Dense>
#include <string>

class Layer
{
private:
	std::vector<SigmoidNeuron> neurons_; // Collection of neurons
	int num_inputs_;                    // Size of input vector
	int num_neurons_;                   // Number of neurons in layer
	Eigen::MatrixXd weights_;           // Weight matrix (num_neurons x num_inputs)
	Eigen::VectorXd biases_;            // Bias vector (num_neurons)
	Eigen::VectorXd activations_;       // Cached activations (num_neurons)

public:
	// Constructor: num_inputs is input size, num_neurons is number of neurons
	Layer(int num_inputs, int num_neurons, unsigned int seed = 42);

	// Forward pass: compute activations for all neurons
	Eigen::VectorXd forward(const Eigen::VectorXd& input);

	// Compute gradients for all neurons
	void compute_gradients(const Eigen::VectorXd& deltas,
		Eigen::MatrixXd& weight_grads,
		Eigen::VectorXd& bias_grads) const;

	// Update parameters for all neurons
	void update_parameters(const Eigen::MatrixXd& weight_grads,
		const Eigen::VectorXd& bias_grads,
		double learning_rate);

	// Print parameters for all neurons in text or JSON format
	std::string print_parameters(bool json_format = false) const;

	// Get activations (for next layer)
	const Eigen::VectorXd& get_activations() const { return activations_; }
};

#endif // LAYER_HPP