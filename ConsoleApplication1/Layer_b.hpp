#pragma once
#ifndef LAYER_B_HPP
#define LAYER_B_HPP

#include"GPUPass.hpp"
#include <random>

//TODO: 
// weights and biases storage will be centralized in network.
// Layer's gonna vanish :-)

class Layer_b {
public:
	Layer_b(int num_inputs, int num_neurons, const Activation* activation, GPUPass* gpuCtx, unsigned int seed = 42);

	~Layer_b();

	//Delete copy and move operations
	Layer_b(const Layer_b&) = delete;
	Layer_b& operator=(const Layer_b&) = delete;
	Layer_b(Layer_b&&) = delete;
	Layer_b& operator=(Layer_b&&) = delete;

public:
	void update_parameters(
		const double* accumulate_weight_grads,
		const double* accumulate_bias_grads,
		double* d_temp_weight_grads, double* d_temp_bias_grads,
		double scale = 1.0);

	// Batched forward on GPU: Computes into provided batch buffers
	// d_batch_input: num_inputs_ x batch_size (column-major)
	// d_batch_z: pre-activations buffer (num_neurons_ x batch_size)
	// d_batch_a: activations buffer (num_neurons_ x batch_size)
	// Returns d_batch_a for chaining
	const double* forward_gpu_batch(const double* d_batch_input, double* d_batch_z, double* d_batch_a, int batch_size);

public:
	const int get_num_neurons() const { return num_neurons_; };
	const int get_num_inputs() const { return num_inputs_; };
	const Eigen::MatrixXd& get_weights() const { return weights_; }
	const Eigen::VectorXd& get_biases() const { return biases_; };
	double* get_d_weights() const { return d_weights_; }
	double* get_d_biases() const { return d_biases_; }
public:
	void set_weights(const Eigen::MatrixXd& weights);
	void set_biases(const Eigen::VectorXd& biases);
private:
	int num_inputs_;                    ///< Number of inputs to the layer
	int num_neurons_;                   ///< Number of neurons in the layer
	Eigen::MatrixXd weights_;           ///< Weight matrix (num_neurons x num_inputs)
	Eigen::VectorXd biases_;            ///< Bias vector (num_neurons)
	double* d_weights_;                 ///< GPU pointer for weights
	double* d_biases_;                  ///< GPU pointer for biases
	std::mt19937 rng_;                  ///< Random number generator for initialization
private:
	GPUPass* gpuCtx_;
	const Activation* activation_;      ///< Pointer to activation function (owned externally)
};


#endif // LAYER_B_HPP