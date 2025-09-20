#include"Layer_b.hpp"



Layer_b::Layer_b(int num_inputs, int num_neurons, const Activation* activation, GPUPass* gpuCtx, unsigned int seed)
	: num_inputs_(num_inputs), num_neurons_(num_neurons),
	weights_(num_neurons, num_inputs), biases_(num_neurons),
	d_weights_(nullptr), d_biases_(nullptr),
	rng_(seed), gpuCtx_(gpuCtx), activation_(activation){

	//TODO: Error check input

	// Xavier initialization
	double stddev = std::sqrt(2.0 / (num_inputs + 1));
	std::normal_distribution<double> dist(0.0, stddev);

	// Initialize weights and biases 
	for (int i = 0; i < num_neurons; ++i) {
		for (int j = 0; j < num_inputs; ++j) {
			weights_(i, j) = dist(rng_);
		}
		biases_(i) = dist(rng_);
	}

	gpuCtx_->allocate_weights(&d_weights_, num_neurons, num_inputs);
	gpuCtx_->allocate_biases(&d_biases_, num_neurons);

	gpuCtx_->copy_weights_to_device(d_weights_, weights_);
	gpuCtx_->copy_biases_to_device(d_biases_, biases_);
}

Layer_b::~Layer_b() {
	gpuCtx_->free_weights(d_weights_);
	gpuCtx_->free_biases(d_biases_);
}

void Layer_b::update_parameters(
	const double* accumulate_weight_grads,
	const double* accumulate_bias_grads,
	double* d_temp_weight_grads,
	double* d_temp_bias_grads,
	double scale) {

	gpuCtx_->updateParametersGPU(
		d_weights_, d_biases_,
		accumulate_weight_grads, accumulate_bias_grads,
		d_temp_weight_grads, d_temp_bias_grads,
		num_neurons_, num_inputs_, num_neurons_, scale
	);

	// Update host weights and biases for compatibility
	gpuCtx_->copy_weights_to_host(weights_, d_weights_, num_neurons_, num_inputs_);
	gpuCtx_->copy_biases_to_host(biases_, d_biases_, num_neurons_);
}

const double* Layer_b::forward_gpu_batch(const double* d_batch_input, double* d_batch_z, double* d_batch_a, int batch_size) {

	// Compute linear: z = W * input + b (batched)
	gpuCtx_->computeLinearGPU_batch(d_weights_, d_batch_input, d_biases_, d_batch_z, num_neurons_, num_inputs_, batch_size);

	// Apply activation
	gpuCtx_->applyActivationGPU_batch(d_batch_z, d_batch_a, num_neurons_, batch_size, activation_);

	// Optionally: Copy last activation to host for single-example compatibility
	// But skip for now to avoid overhead
	return d_batch_a;
}

void Layer_b::set_weights(const Eigen::MatrixXd& weights)
{
	assert(weights.rows() == weights_.rows() && weights.cols() == weights_.cols());
	weights_ = weights;
	gpuCtx_->copy_weights_to_device(d_weights_, weights_);
}

void Layer_b::set_biases(const Eigen::VectorXd& biases) {
	assert(biases.size() == biases_.size());
	biases_ = biases;
	gpuCtx_->copy_biases_to_device(d_biases_, biases_);
}