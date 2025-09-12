#pragma once
#ifndef LAYER_HPP
#define LAYER_HPP

#include <vector>
#include <Eigen/Dense>
#include <string>
#include <random>
#include "Activation.hpp"  
#include "ComputationContext.hpp"

// Forward declarations
class GPUComputationContext;
class CPUComputationContext;

/**
 * @brief A neural network layer with sigmoid activation.
 *
 * Manages a layer of neurons with weights, biases, and activations,
 * performing forward propagation, gradient computation, and parameter updates.
 */

class Layer
{
public:
	Layer(int num_inputs, int num_neurons, const Activation* activation, ComputationContext* context, unsigned int seed = 42);

	~Layer();

	//Delete copy and move operations
	Layer(const Layer&) = delete;
	Layer& operator=(const Layer&) = delete;
	Layer(Layer&&) = delete;
	Layer& operator=(Layer&&) = delete;

public:
	Eigen::VectorXd forward_cpu(const Eigen::VectorXd& input);
	const double* forward_gpu(const double* input);

	void compute_gradients_cpu(
		const Eigen::VectorXd& deltas,
		Eigen::MatrixXd& weight_grads,
		Eigen::VectorXd& bias_grads, bool apply_derivative = true) const;

	void compute_gradients_gpu(double* d_deltas, bool apply_derivative = true);

	void update_parameters_cpu(
		const Eigen::MatrixXd& weight_grads,
		const Eigen::VectorXd& bias_grads, 
		double scale = 1.0);

	void update_parameters_gpu(
		const double* accumulate_weight_grads,
		const double* accumulate_bias_grads,
		double* d_temp_weight_grads, double* d_temp_bias_grads,
		double scale = 1.0);

	std::string print_parameters(bool json_format = false) const;

	// New GPU helper: Apply derivative elementwise on device
	void apply_derivative_gpu(double* d_delta);

public:
	const Eigen::VectorXd& get_activations() const { return activations_; }
	const Eigen::VectorXd& get_pre_activations() const { return pre_activations_; } // New getter
	const Eigen::MatrixXd& get_weights() const { return weights_; }
	const Eigen::VectorXd& get_biases() const { return biases_; };
	const int get_num_neurons() const { return num_neurons_; };
	const int get_num_inputs() const { return num_inputs_; };

	double* get_d_input_() const { return d_input_; }
	double* get_d_pre_activations_() const { return d_pre_activations_; }
	double* get_d_activations_() const { return d_activations_; }
	double* get_d_weights() const { return d_weights_; }
	double* get_d_biases() const { return d_biases_; }
	double* get_d_derivatives() const { return d_derivatives; }
	double* get_d_dy() const { return d_dy; }
	double* get_d_weight_grads_() const { return d_weight_grads_; }  // New
	//double* get_d_bias_grads_() const { return d_bias_grads_; }      // New
	double* get_d_delta_() const { return d_delta_; }  

	void set_weights(const Eigen::MatrixXd& weights);
	void set_biases(const Eigen::VectorXd& biases);
	void set_pre_activations(const Eigen::VectorXd& pre_activations);
	void set_activations(const Eigen::VectorXd& activations);

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
	const Activation* activation_;      ///< Pointer to activation function (owned externally)
	ComputationContext* context_;       ///< Computation context for operations
	GPUComputationContext* contextGPU_ = nullptr;
	CPUComputationContext* contextCPU_ = nullptr;
	bool is_gpu_context_;               // New: Cached flag
	double* d_weights_;                 ///< GPU pointer for weights
	double* d_biases_;                  ///< GPU pointer for biases
	double* d_input_;
	double* d_pre_activations_;
	double* d_activations_;
	double* d_derivatives;
	double* d_dy;
	double* d_weight_grads_;            ///< GPU pointer for weight gradients
	double* d_delta_;					///< GPU pointer for delta (and bias gradients)
	double* d_temp_;					///< GPU temp for adjusted delta
};

#endif // LAYER_HPP