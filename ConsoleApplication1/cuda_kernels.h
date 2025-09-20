#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include <cuda_runtime.h>

namespace cuda_kernels {
	//CUDA kernel for cross-entropy loss
	__global__ void crossEntropyLossKernel(const double* output, const double* target, double* loss, int n);

	// CUDA kernel for sum reduction
	__global__ void sumReductionKernel(const double* input, double* output, int n);

	// New kernel for broadcasting biases across batch
	// Beginner note: For a batch, we need to add biases (size num_neurons) to each column of the output matrix
	// (num_neurons × batch_size). This kernel sets z[i, j] += biases[i] for each batch index j.
	__global__ void add_bias_batch(double* z, const double* biases, int m, int batch_size);

	// Kernel for batched elementwise subtract
	__global__ void elementwise_subtract_batch_kernel(const double* a, const double* b, double* c, int rows, int batch_size);

	// Kernel for batched elementwise multiply
	__global__ void elementwise_multiply_batch_kernel(const double* a, const double* b, double* c, int rows, int batch_size);

	// Kernel for batched sigmoid derivative
	__global__ void sigmoid_prime_batch_kernel(const double* z, double* out, int rows, int batch_size);

	// New kernel for setting array to zero
	__global__ void set_to_zero_kernel(double* data, int n);

	// CUDA kernel for element-wise multiplication
	__global__ void elementwise_multiply(const double* a, const double* b, double* c, int n);

	//debug kernel
	__global__ void debugPrint_kernel(const double* data, int n);
}


#endif