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
    void updateParameters(Eigen::MatrixXd& weights,
        Eigen::VectorXd& biases,
        const Eigen::MatrixXd& weight_grads,
        const Eigen::VectorXd& bias_grads, double scale) override;

    void accumulateGradients(const std::vector<Eigen::MatrixXd>& weight_grads_in,
        const std::vector<Eigen::VectorXd>& bias_grads_in,
        std::vector<Eigen::MatrixXd>& weight_grads_out,
        std::vector<Eigen::VectorXd>& bias_grads_out,
        double scale) override;
};

// Optional: Factory function to create an instance
// GPUComputationContext* createGPUContext() {
//     return new GPUComputationContext();
// }