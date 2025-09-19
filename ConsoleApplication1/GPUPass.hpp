#pragma once

#include "Activation.hpp"
#include <Eigen/Dense>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <iostream>

// Error checking macros for CUDA, cuBLAS, and cuDNN
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
}

#define CHECK_CUBLAS(call) { \
    cublasStatus_t stat = call; \
    if (stat != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error in " << __FILE__ << ":" << __LINE__ << ": " \
                  << stat << std::endl; \
        exit(1); \
    } \
}

#define CHECK_CUDNN(call) { \
    cudnnStatus_t stat = call; \
    if (stat != CUDNN_STATUS_SUCCESS) { \
        std::cerr << "cuDNN error in " << __FILE__ << ":" << __LINE__ << ": " \
                  << cudnnGetErrorString(stat) << std::endl; \
        exit(1); \
    } \
}

class GPUPass {
public:
    // Constructor initializes cuBLAS and cuDNN handles
    GPUPass() {
        CHECK_CUBLAS(cublasCreate(&cublasHandle));
        CHECK_CUDNN(cudnnCreate(&cudnnHandle));
    }

    // Destructor cleans up handles
    ~GPUPass() {
        cublasDestroy(cublasHandle);
        cudnnDestroy(cudnnHandle);
    }

public:
    //NonBatched funcs to be ported
    double compute_mse_lossGPU(const Eigen::VectorXd& output, const Eigen::VectorXd& target);
    double compute_cross_entropy_lossGPU(const Eigen::VectorXd& output, const Eigen::VectorXd& target);
    
public:
    //common batched-single example
    void updateParametersGPU(
        double* d_weights, double* d_biases,
        const double* weight_grads, const double* bias_grads,
        double* d_temp_weight_grads, double* d_temp_bias_grads,
        int m, int n, int bias_size, double scale);

    double compute_gradient_norm_gpu(
        const std::vector<double*>& weight_grads, const std::vector<double*>& bias_grads,
        const std::vector<int>& w_rows, const std::vector<int>& w_cols, const std::vector<int>& b_sizes, size_t batch_size);

    void add_regularization(double* d_weight_grad, double* d_weights, double scale, int m, int n);
    double compute_squared_norm_gpu(double* d_data, int n);
public:
    //batched funcs

    // Placeholder function to dynamically estimate max batch size based on GPU memory.
    int get_dynamic_max_batch_size(int approx_network_size = 10000) const {
        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));  // Get properties of device 0 (default GPU)
        size_t total_mem = prop.totalGlobalMem;  // Total GPU memory in bytes
        size_t available_mem = total_mem * 0.1;  // Conservatively use 10% to avoid OOM
        size_t mem_per_example = sizeof(double) * approx_network_size * 2;  // Rough estimate: inputs + outputs per layer
        return static_cast<int>(available_mem / mem_per_example);
    }

    // Placeholder for sub-batching: Process large batches in chunks
    void process_subbatched(std::function<void(int sub_batch_size)> func, int total_batch_size, int max_batch);

    // Compute linear transformation for a batch: z = W * X + b
    // W: num_neurons × num_inputs, X: num_inputs × batch_size, b: num_neurons, z: num_neurons × batch_size
    void computeLinearGPU_batch(const double* d_weights, const double* d_batch_input, const double* d_biases,
        double* d_batch_z, int m, int n, int batch_size);

    // Apply activation function to a batch matrix (e.g., sigmoid on each element)
    void applyActivationGPU_batch(const double* d_batch_z, double* d_batch_a, int vec_size, int batch_size,
        const Activation* activation);

    // Batched elementwise subtract: c = a - b (rows x batch_size, column-major)
    void launch_elementwise_subtract_batch(const double* a, const double* b, double* c, int rows, int batch_size);

    // Batched elementwise multiply: c = a * b (rows x batch_size)
    void launch_elementwise_multiply_batch(const double* a, const double* b, double* c, int rows, int batch_size);

    // Batched gradient computation: Accumulates weight_grad += delta * prev_a^T, bias_grad += sum(delta)
    void computeGradientsGPU_batch(const double* d_deltas_batch, const double* d_prev_activations_batch, double* d_weight_grads, double* d_bias_grads, int m, int n, int batch_size);

    // Batched delta propagation: delta = W^T * next_delta (n x batch_size)
    void compute_delta_back_batch(const double* d_weights, const double* d_delta_next_batch, double* d_delta_batch, int m, int n, int batch_size);

    // Batched activation derivative: derivatives = activation'(pre_activations) (vec_size x batch_size)
    void computeActivationDerivativeGPU_batch(const double* d_pre_activations, double* d_derivatives, int vec_size, int batch_size, const Activation* activation);

    //computes cost prime for mse or crossentropy with sigmoid. 
    void cost_prime_mse_crossent_batched(const double* d_output, const double* d_target, double* d_delta, int rows, int batch_size);
public:
    // Memory management for GPU
    void allocate_weights(double** d_weights, int rows, int cols);
    void allocate_biases(double** d_biases, int size);
    void copy_weights_to_device(double* d_weights, const Eigen::MatrixXd& weights);
    void copy_biases_to_device(double* d_biases, const Eigen::VectorXd& biases);
    void copy_weights_to_host(Eigen::MatrixXd& weights, double* d_weights, int rows, int cols);
    void copy_biases_to_host(Eigen::VectorXd& biases, double* d_biases, int size);
    void free_weights(double* d_weights);
    void free_biases(double* d_biases);

public:
    void allocate_batch_vector(double** d_vec, int vec_size, int batch_size);
    void free_batch_vector(double* d_vec);
    void copy_batch_to_device(double* d_batch_matrix, const std::vector<Eigen::VectorXd>& batch, bool transpose = false);
    void copy_batch_to_host(std::vector<Eigen::VectorXd>& batch, const double* d_batch_matrix, int vec_size, int batch_size);
    void set_to_zero_batch(double* d_data, int size, int batch_size);
public:
    void launch_elementwise_multiply(const double* a, const double* b, double* c, int n);
    void launch_elementwise_subtract(const double* a, const double* b, double* c, int n);
    void debugPrint(const double* data, int n);
    void debugPrint_batch(const double* d_data, int rows, int cols);

private:
    cublasHandle_t cublasHandle; // Handle for cuBLAS operations
    cudnnHandle_t cudnnHandle;   // Handle for cuDNN operations
};