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
	Eigen::VectorXd pre_activations_;   // Cached pre-activations (z = W * a + b)
	Eigen::VectorXd input_;             // Cached input for the layer

private:
	// Sigmoid activation function
	static double sigmoid(double x);

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
		const Eigen::VectorXd& bias_grads);

	// Print parameters for all neurons in text or JSON format
	std::string print_parameters(bool json_format = false) const;

	// Get activations (for next layer)
	const Eigen::VectorXd& get_activations() const { return activations_; }
	const Eigen::VectorXd& get_pre_activations() const { return pre_activations_; } // New getter
	const Eigen::MatrixXd& get_weights() const { return weights_; }
	const Eigen::VectorXd& get_biases() const { return biases_; };
	const int get_num_neurons() const { return num_neurons_; };
	const int get_num_inputs() const { return num_inputs_; };
};

#endif // LAYER_HPP