#include "ComputationContext.hpp"
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


// GPU implementation of ComputationContext using CUDA, cuBLAS, and cuDNN
class GPUComputationContext : public ComputationContext {
private:
    cublasHandle_t cublasHandle; // Handle for cuBLAS operations
    cudnnHandle_t cudnnHandle;   // Handle for cuDNN operations
public:
    // Constructor initializes cuBLAS and cuDNN handles
    GPUComputationContext() {
        CHECK_CUBLAS(cublasCreate(&cublasHandle));
        CHECK_CUDNN(cudnnCreate(&cudnnHandle));
    }

    // Destructor cleans up handles
    ~GPUComputationContext() {
        cublasDestroy(cublasHandle);
        cudnnDestroy(cudnnHandle);
    }

    void fuckingHell() override;
public:

    // Compute linear transformation on GPU: weights * input + biases
    void computeLinearGPU(const double* d_weights,const double* d_input,const double* d_biases, double* d_z, int m, int n);

    // Apply activation function (assumes sigmoid for now using cuDNN)
    void applyActivationGPU(double* d_z, double* d_a, int n, const Activation* activation);

    // Compute activation derivative 
    Eigen::VectorXd computeActivationDerivativeGPU(
        double* d_a, double* d_z,
        double* d_dy, double* d_derivatives,
        int size, const Activation* activation);

    // Compute gradients on GPU
    void computeGradientsGPU(
        double* d_incoming_deltas,   // incoming deltas (on device)
        double* d_input,             // input vector (on device)
        double* d_derivatives,       // activation derivatives (on device, computed beforehand if needed)
        double* d_weight_grads,      // output: weight gradients
        double* d_bias_grads,        // output: bias gradients
        double* d_temp,              // temp buffer, same size as num_neurons
        int num_neurons,
        int num_inputs,
        bool apply_derivative);

    void updateParametersGPU(
        double* d_weights,
        double* d_biases,
        const double* weight_grads,
        const double* bias_grads,
        double* d_temp_weight_grads, 
        double* d_temp_bias_grads,
        int m, int n, int bias_size, double scale);

    void accumulateGradientsGPU(
        const std::vector<double*>& weight_grads_in,
        const std::vector<double*>& bias_grads_in,
        const std::vector<double*>& weight_grads_out,
        const std::vector<double*>& bias_grads_out,
        const std::vector<int>& weight_rows,
        const std::vector<int>& weight_cols,
        const std::vector<int>& bias_sizes,
        double scale);

public:
    double compute_mse_lossGPU(const Eigen::VectorXd& output, const Eigen::VectorXd& target);
    double compute_cross_entropy_lossGPU(const Eigen::VectorXd& output, const Eigen::VectorXd& target);
    double compute_gradient_norm_gpu(
        const std::vector<double*>& weight_grads, const std::vector<double*>& bias_grads,
        const std::vector<int>& w_rows, const std::vector<int>& w_cols, const std::vector<int>& b_sizes, size_t batch_size);

    void add_regularization(double* d_weight_grad, double* d_weights, double scale, int m, int n);
    void compute_delta_back(double* d_weights, double* d_delta_next, double* d_delta, int m, int n);

    double compute_squared_normGPU(const Eigen::MatrixXd& matrix);
    double compute_squared_norm_gpu(double* d_data, int n);

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

    // methods to support GPU memory allocation and operations.
    void allocate_vector(double** d_vector, int size);
    void free_vector(double* d_vector);
    void copy_to_device(double* d_vector, const Eigen::VectorXd& vector);
    void copy_to_device(double* d_matrix, const Eigen::MatrixXd& matrix);
    void copy_to_host(Eigen::VectorXd& vector, double* d_vector, int size);
    void copy_to_host(Eigen::MatrixXd& matrix, double* d_matrix, int rows, int cols);
    void set_to_zero(double* d_data, int n);
    void copy_device_to_device(double* dst, const double* src, int size);

public:
    void launch_elementwise_multiply(const double* a, const double* b, double* c, int n);
    void launch_elementwise_subtract(const double* a, const double* b, double* c, int n);
    //debug
    void debugPrint(const double* data, int n);
};
