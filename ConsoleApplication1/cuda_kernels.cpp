#ifdef __INTELLISENSE__
#define CUDA_KERNEL_NODE_PARAMS
#define __CUDACC__
#endif

#include"cuda_kernels.h"
#include<iostream>

namespace cuda_kernels {
    //CUDA kernel for cross-entropy loss
    __global__ void crossEntropyLossKernel(const double* output, const double* target, double* loss, int n)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            double a = max(1e-15, min(1.0 - 1e-15, output[idx]));
            loss[idx] = -(target[idx] * log(a) + (1.0 - target[idx]) * log(1.0 - a));
        }
    }

    // CUDA kernel for sum reduction
    __global__ void sumReductionKernel(const double* input, double* output, int n)
    {
        extern __shared__ double sdata[];
        unsigned int tid = threadIdx.x;
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

        sdata[tid] = (idx < n) ? input[idx] : 0.0;
        __syncthreads();

        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            output[blockIdx.x] = sdata[0];
        }
    }

    // New kernel for broadcasting biases across batch
    // Beginner note: For a batch, we need to add biases (size num_neurons) to each column of the output matrix
    // (num_neurons × batch_size). This kernel sets z[i, j] += biases[i] for each batch index j.
    __global__ void add_bias_batch(double* z, const double* biases, int m, int batch_size) {
        int row = blockIdx.x * blockDim.x + threadIdx.x; // Row index (neuron)
        int col = blockIdx.y * blockDim.y + threadIdx.y; // Column index (batch element)
        if (row < m && col < batch_size) {
            int idx = row + col * m; // Column-major index: z[row, col] = z[row + col * m]
            z[idx] += biases[row];
        }
    }

    // Kernel for batched elementwise subtract
    __global__ void elementwise_subtract_batch_kernel(const double* a, const double* b, double* c, int rows, int batch_size) {
        int row = blockIdx.x * blockDim.x + threadIdx.x;  // Neuron index
        int col = blockIdx.y * blockDim.y + threadIdx.y;  // Batch item index
        if (row < rows && col < batch_size) {
            int idx = row + col * rows;  // Column-major
            c[idx] = a[idx] - b[idx];
        }
    }

    // Kernel for batched elementwise multiply
    __global__ void elementwise_multiply_batch_kernel(const double* a, const double* b, double* c, int rows, int batch_size) {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        int col = blockIdx.y * blockDim.y + threadIdx.y;
        if (row < rows && col < batch_size) {
            int idx = row + col * rows;
            c[idx] = a[idx] * b[idx];
        }
    }

    // Kernel for batched sigmoid derivative
    __global__ void sigmoid_prime_batch_kernel(const double* z, double* out, int rows, int batch_size) {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        int col = blockIdx.y * blockDim.y + threadIdx.y;
        if (row < rows && col < batch_size) {
            int idx = row + col * rows;
            double sig = 1.0 / (1.0 + exp(-z[idx]));
            out[idx] = sig * (1.0 - sig);
        }
    }

    // New kernel for setting array to zero
    __global__ void set_to_zero_kernel(double* data, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            data[idx] = 0.0;
        }
    }

    // CUDA kernel for element-wise multiplication
    __global__ void elementwise_multiply(const double* a, const double* b, double* c, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            c[idx] = a[idx] * b[idx];
        }
    }

    //debug kernel
    __global__ void debugPrint_kernel(const double* data, int n) {
        /*int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            printf("data[%d] = %f\n", idx, data[idx]);
        }*/

        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < n) printf("data[%d] = %f\n", idx, data[idx]);

    }
}

