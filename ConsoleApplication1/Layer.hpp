#pragma once
#ifndef LAYER_HPP
#define LAYER_HPP

#include <vector>
#include <Eigen/Dense>
#include <string>
#include <random>

/**
 * @brief A neural network layer with sigmoid activation.
 *
 * Manages a layer of neurons with weights, biases, and activations,
 * performing forward propagation, gradient computation, and parameter updates.
 */

class Layer
{
private:
	int num_inputs_;                    ///< Number of inputs to the layer
	int num_neurons_;                   ///< Number of neurons in the layer
	Eigen::MatrixXd weights_;           ///< Weight matrix (num_neurons x num_inputs)
	Eigen::VectorXd biases_;            ///< Bias vector (num_neurons)
	Eigen::VectorXd activations_;       ///< Cached activations (num_neurons)
	Eigen::VectorXd pre_activations_;   ///< Cached pre-activations (z = W * a + b)
	Eigen::VectorXd input_;             ///< Cached input for gradient computation
	std::mt19937 rng_;                  ///< Random number generator for initialization
	bool has_valid_activations_;        ///< Tracks if activations are valid

private:
	/**
	 * @brief Sigmoid activation function.
	 * @param x Input value
	 * @return Sigmoid of x (1 / (1 + exp(-x)))
	 */
	static double sigmoid(double x);

public:
	/**
	 * @brief Constructs a layer with specified input size and number of neurons.
	 * @param num_inputs Size of input vector
	 * @param num_neurons Number of neurons in the layer
	 * @param seed Random seed for weight/bias initialization
	 */
	Layer(int num_inputs, int num_neurons, unsigned int seed = 42);

	/**
	 * @brief Computes the forward pass, producing activations for the input.
	 * @param input Input vector (num_inputs x 1)
	 * @return Output activations (num_neurons x 1)
	 */
	Eigen::VectorXd forward(const Eigen::VectorXd& input);

	/**
	 * @brief Computes gradients for weights and biases.
	 * @param deltas Error terms from the next layer or cost function
	 * @param weight_grads Output weight gradients (num_neurons x num_inputs)
	 * @param bias_grads Output bias gradients (num_neurons)
	 */
	void compute_gradients(const Eigen::VectorXd& deltas,
		Eigen::MatrixXd& weight_grads,
		Eigen::VectorXd& bias_grads) const;

	/**
	 * @brief Updates weights and biases using pre-computed gradients.
	 * @param weight_grads Weight gradients (num_neurons x num_inputs)
	 * @param bias_grads Bias gradients (num_neurons)
	 */
	void update_parameters(const Eigen::MatrixXd& weight_grads,
		const Eigen::VectorXd& bias_grads);

	/**
	 * @brief Prints layer parameters (weights, biases, activations).
	 * @param json_format If true, output in JSON-like format; else, text format
	 * @return String representation of parameters
	 */
	std::string print_parameters(bool json_format = false) const;

	/**
	 * @brief Gets the current activations.
	 * @return Reference to activations vector
	 */
	const Eigen::VectorXd& get_activations() const { return activations_; }

	/**
	 * @brief Gets the pre-activation values (z = W * a + b).
	 * @return Reference to pre-activations vector
	 */
	const Eigen::VectorXd& get_pre_activations() const { return pre_activations_; } // New getter
	
	/**
	 * @brief Gets the weight matrix.
	 * @return Reference to weights matrix
	 */
	const Eigen::MatrixXd& get_weights() const { return weights_; }

	/**
	 * @brief Gets the bias vector.
	 * @return Reference to biases vector
	 */
	const Eigen::VectorXd& get_biases() const { return biases_; };

	/**
	 * @brief Gets the number of neurons.
	 * @return Number of neurons
	 */
	const int get_num_neurons() const { return num_neurons_; };

	/**
	 * @brief Gets the number of inputs.
	 * @return Number of inputs
	 */
	const int get_num_inputs() const { return num_inputs_; };
};

#endif // LAYER_HPP