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

    // Compute linear transformation on GPU: weights * input + biases
    Eigen::VectorXd computeLinear(const Eigen::MatrixXd& weights,
        const Eigen::VectorXd& input,
        const Eigen::VectorXd& biases) override;

    // New method for computing weight gradients (outer product)
    Eigen::MatrixXd computeWeightGradient(const Eigen::VectorXd& delta,
        const Eigen::VectorXd& activation) override;

    // Apply activation function (assumes sigmoid for now using cuDNN)
    Eigen::VectorXd applyActivation(const Eigen::VectorXd& z,
        const Activation* activation) override;

    // Compute activation derivative 
    Eigen::VectorXd computeActivationDerivative(const Eigen::VectorXd& activations,
        const Eigen::VectorXd& pre_activations,
        const Activation* activation) override;

    // Compute gradients on GPU
    void computeGradients(const Eigen::VectorXd& deltas,
        const Eigen::VectorXd& activation_derives,
        const Eigen::VectorXd& input,
        Eigen::MatrixXd& weight_grads,
        Eigen::VectorXd& bias_grads) override;

    // Update parameters (on host for now; optimize later)
    void updateParameters(
        Eigen::MatrixXd& weights,
        Eigen::VectorXd& biases,
        const Eigen::MatrixXd& weight_grads,
        const Eigen::VectorXd& bias_grads, double scale) override;

    void accumulateGradients(
        const std::vector<Eigen::MatrixXd>& weight_grads_in,
        const std::vector<Eigen::VectorXd>& bias_grads_in,
        std::vector<Eigen::MatrixXd>& weight_grads_out,
        std::vector<Eigen::VectorXd>& bias_grads_out,
        double scale) override;

    double compute_squared_norm(const Eigen::MatrixXd& matrix) override;
    double compute_mse_loss(const Eigen::VectorXd& output, const Eigen::VectorXd& target) override;
    double compute_cross_entropy_loss(const Eigen::VectorXd& output, const Eigen::VectorXd& target) override;

    // Memory management for GPU
    void allocate_weights(double** d_weights, int rows, int cols) override;
    void allocate_biases(double** d_biases, int size) override;
    void copy_weights_to_device(double* d_weights, const Eigen::MatrixXd& weights) override;
    void copy_biases_to_device(double* d_biases, const Eigen::VectorXd& biases) override;
    void copy_weights_to_host(Eigen::MatrixXd& weights, double* d_weights, int rows, int cols) override;
    void copy_biases_to_host(Eigen::VectorXd& biases, double* d_biases, int size) override;
    void free_weights(double* d_weights) override;
    void free_biases(double* d_biases) override;

    // methods to support GPU memory allocation and operations.
    void allocate_vector(double** d_vector, int size) override;
    void free_vector(double* d_vector) override;
    void copy_to_device(double* d_vector, const Eigen::VectorXd& vector) override;
    void copy_to_device(double* d_matrix, const Eigen::MatrixXd& matrix) override;
    void copy_to_host(Eigen::VectorXd& vector, double* d_vector, int size) override;
    void copy_to_host(Eigen::MatrixXd& matrix, double* d_matrix, int rows, int cols) override;
    void computeLinearGPU(double* d_weights, double* d_input, double* d_biases, double* d_z, int m, int n) override;
    void applyActivationGPU(double* d_z, double* d_a, int n, const Activation* activation) override;

    Eigen::VectorXd computeActivationDerivativeGPU(
        double* d_a, double* d_z, 
        double* d_dy, double* d_derivatives, 
        int size, const Activation* activation) override;

    void computeGradientsGPU(
        const Eigen::VectorXd& deltas,
        double* d_derivatives,
        double* d_input,
        double* weight_grads,
        double* bias_grads,
        int m, int n) override;

    void updateParametersGPU(
        double* d_weights,
        double* d_biases,
        double* weight_grads,
        double* bias_grads,
        int m, int n, int bias_size, double scale) override;

    void accumulateGradientsGPU(
        const std::vector<double*>& weight_grads_in,
        const std::vector<double*>& bias_grads_in,
        const std::vector<double*>& weight_grads_out,
        const std::vector<double*>& bias_grads_out,
        const std::vector<int>& weight_rows,
        const std::vector<int>& weight_cols,
        const std::vector<int>& bias_sizes,
        double scale) override;

    //debug
    void debugPrint(const double* data, int n) override;
};

// Optional: Factory function to create an instance
// GPUComputationContext* createGPUContext() {
//     return new GPUComputationContext();
// }