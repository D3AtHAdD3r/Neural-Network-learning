#include"GPUComputationContext.hpp"

Eigen::VectorXd GPUComputationContext::computeLinear(const Eigen::MatrixXd& weights, const Eigen::VectorXd& input, const Eigen::VectorXd& biases)
{
    int m = weights.rows(); // Output size
    int n = weights.cols(); // Input size

    // Device memory pointers
    double* d_weights, * d_input, * d_biases, * d_z;

    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&d_weights, m * n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_input, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_biases, m * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_z, m * sizeof(double)));

    // Transfer data to device
    CHECK_CUDA(cudaMemcpy(d_weights, weights.data(), m * n * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_input, input.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_biases, biases.data(), m * sizeof(double), cudaMemcpyHostToDevice));

    // Compute z = weights * input using cuBLAS
    double alpha = 1.0, beta = 0.0;
    CHECK_CUBLAS(cublasDgemv(cublasHandle, CUBLAS_OP_N, m, n, &alpha, d_weights, m,
        d_input, 1, &beta, d_z, 1));

    // Add biases: z = z + biases
    alpha = 1.0;
    CHECK_CUBLAS(cublasDaxpy(cublasHandle, m, &alpha, d_biases, 1, d_z, 1));

    // Transfer result back to host
    Eigen::VectorXd z(m);
    CHECK_CUDA(cudaMemcpy(z.data(), d_z, m * sizeof(double), cudaMemcpyDeviceToHost));

    // Free device memory
    CHECK_CUDA(cudaFree(d_weights));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_biases));
    CHECK_CUDA(cudaFree(d_z));

    return z;
}

Eigen::VectorXd GPUComputationContext::applyActivation(const Eigen::VectorXd& z, const Activation* activation) {
    if (!activation) {
        throw std::runtime_error("Null activation pointer in applyActivation");
    }

    int n = z.size();
    if (n == 0) {
        throw std::runtime_error("Empty input vector in applyActivation");
    }

    // Get cuDNN activation mode
    cudnnActivationMode_t mode = activation->getCudnnActivationMode();

    // Device memory pointers
    double* d_z, * d_a;

    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&d_z, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_a, n * sizeof(double)));

    // Transfer pre-activations to device
    CHECK_CUDA(cudaMemcpy(d_z, z.data(), n * sizeof(double), cudaMemcpyHostToDevice));

    // Set up cuDNN descriptors
    cudnnTensorDescriptor_t tensorDesc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&tensorDesc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(tensorDesc, CUDNN_TENSOR_NCHW,
        CUDNN_DATA_DOUBLE, 1, 1, n, 1));

    cudnnActivationDescriptor_t activationDesc;
    CHECK_CUDNN(cudnnCreateActivationDescriptor(&activationDesc));
    CHECK_CUDNN(cudnnSetActivationDescriptor(activationDesc, mode,
        CUDNN_NOT_PROPAGATE_NAN, 0.0));

    // Apply activation on GPU
    double alpha = 1.0, beta = 0.0;
    CHECK_CUDNN(cudnnActivationForward(cudnnHandle, activationDesc, &alpha, tensorDesc,
        d_z, &beta, tensorDesc, d_a));

    // Transfer activations back to host
    Eigen::VectorXd a(n);
    CHECK_CUDA(cudaMemcpy(a.data(), d_a, n * sizeof(double), cudaMemcpyDeviceToHost));

    // Clean up
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(tensorDesc));
    CHECK_CUDNN(cudnnDestroyActivationDescriptor(activationDesc));
    CHECK_CUDA(cudaFree(d_z));
    CHECK_CUDA(cudaFree(d_a));

    return a;
}

Eigen::VectorXd GPUComputationContext::computeActivationDerivative(const Eigen::VectorXd& activations, const Eigen::VectorXd& pre_activations, const Activation* activation) {
    if (!activation) {
        throw std::runtime_error("Null activation pointer in computeActivationDerivative");
    }
    if (activations.size() != pre_activations.size()) {
        throw std::runtime_error("Mismatched sizes in computeActivationDerivative");
    }
    if (activations.size() == 0) {
        throw std::runtime_error("Empty input vectors in computeActivationDerivative");
    }

    int n = activations.size();

    // Get cuDNN activation mode
    cudnnActivationMode_t mode = activation->getCudnnActivationMode();

    // Device memory pointers
    double* d_activations, * d_pre_activations, * d_dy, * d_derivatives;

    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&d_activations, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_pre_activations, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_dy, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_derivatives, n * sizeof(double)));

    // Transfer inputs to device
    CHECK_CUDA(cudaMemcpy(d_activations, activations.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_pre_activations, pre_activations.data(), n * sizeof(double), cudaMemcpyHostToDevice));

    // Initialize dy with ones (to compute raw derivative, not scaled by deltas)
    std::vector<double> ones(n, 1.0);
    CHECK_CUDA(cudaMemcpy(d_dy, ones.data(), n * sizeof(double), cudaMemcpyHostToDevice));

    // Set up cuDNN descriptors
    cudnnTensorDescriptor_t tensorDesc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&tensorDesc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(tensorDesc, CUDNN_TENSOR_NCHW,
        CUDNN_DATA_DOUBLE, 1, 1, n, 1));

    cudnnActivationDescriptor_t activationDesc;
    CHECK_CUDNN(cudnnCreateActivationDescriptor(&activationDesc));
    CHECK_CUDNN(cudnnSetActivationDescriptor(activationDesc, mode,
        CUDNN_NOT_PROPAGATE_NAN, 0.0));

    // Compute derivatives using cuDNN
    double alpha = 1.0, beta = 0.0;
    CHECK_CUDNN(cudnnActivationBackward(cudnnHandle, activationDesc, &alpha, tensorDesc,
        d_activations, tensorDesc, d_dy, tensorDesc, d_pre_activations,
        &beta, tensorDesc, d_derivatives));

    // Transfer result back to host
    Eigen::VectorXd derivatives(n);
    CHECK_CUDA(cudaMemcpy(derivatives.data(), d_derivatives, n * sizeof(double), cudaMemcpyDeviceToHost));

    // Clean up
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(tensorDesc));
    CHECK_CUDNN(cudnnDestroyActivationDescriptor(activationDesc));
    CHECK_CUDA(cudaFree(d_activations));
    CHECK_CUDA(cudaFree(d_pre_activations));
    CHECK_CUDA(cudaFree(d_dy));
    CHECK_CUDA(cudaFree(d_derivatives));

    return derivatives;
}


void GPUComputationContext::computeGradients(const Eigen::VectorXd& deltas, const Eigen::VectorXd& activation_derives, const Eigen::VectorXd& input, Eigen::MatrixXd& weight_grads, Eigen::VectorXd& bias_grads)
{
    int m = deltas.size(); // Output size
    int n = input.size();  // Input size

    // Device memory pointers
    double* d_deltas, * d_deriv, * d_input, * d_adjusted_deltas, * d_weight_grads;

    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&d_deltas, m * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_deriv, m * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_input, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_adjusted_deltas, m * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_weight_grads, m * n * sizeof(double)));

    // Transfer data to device
    CHECK_CUDA(cudaMemcpy(d_deltas, deltas.data(), m * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_deriv, activation_derives.data(), m * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_input, input.data(), n * sizeof(double), cudaMemcpyHostToDevice));

    // Compute adjusted_deltas = deltas .* activation_derives
    // Note: cuBLAS doesn't have direct element-wise multiply; use cublasDdgmm
    double alpha = 1.0;
    CHECK_CUBLAS(cublasDdgmm(cublasHandle, CUBLAS_SIDE_LEFT, m, 1, d_deltas, m,
        d_deriv, 1, d_adjusted_deltas, m));

    // Compute weight_grads = adjusted_deltas * input.transpose()
    alpha = 1.0;
    double beta = 0.0;
    CHECK_CUBLAS(cublasDger(cublasHandle, m, n, &alpha, d_adjusted_deltas, 1,
        d_input, 1, d_weight_grads, m));

    // Bias grads are the adjusted deltas
    bias_grads.resize(m);
    CHECK_CUDA(cudaMemcpy(bias_grads.data(), d_adjusted_deltas, m * sizeof(double), cudaMemcpyDeviceToHost));

    // Transfer weight_grads back to host
    weight_grads.resize(m, n);
    CHECK_CUDA(cudaMemcpy(weight_grads.data(), d_weight_grads, m * n * sizeof(double), cudaMemcpyDeviceToHost));

    // Free device memory
    CHECK_CUDA(cudaFree(d_deltas));
    CHECK_CUDA(cudaFree(d_deriv));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_adjusted_deltas));
    CHECK_CUDA(cudaFree(d_weight_grads));
}

void GPUComputationContext::updateParameters(Eigen::MatrixXd& weights, Eigen::VectorXd& biases, const Eigen::MatrixXd& weight_grads, const Eigen::VectorXd& bias_grads, double scale)
{
    int m = weights.rows(); // Output size
    int n = weights.cols(); // Input size
    int bias_size = biases.size();

    // Device memory pointers
    double* d_weights, * d_biases, * d_weight_grads, * d_bias_grads;

    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&d_weights, m * n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_biases, bias_size * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_weight_grads, m * n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_bias_grads, bias_size * sizeof(double)));

    // Transfer data to device
    CHECK_CUDA(cudaMemcpy(d_weights, weights.data(), m * n * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_biases, biases.data(), bias_size * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_weight_grads, weight_grads.data(), m * n * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_bias_grads, bias_grads.data(), bias_size * sizeof(double), cudaMemcpyHostToDevice));

    // Scale gradients: weight_grads *= scale, bias_grads *= scale
    CHECK_CUBLAS(cublasDscal(cublasHandle, m * n, &scale, d_weight_grads, 1));
    CHECK_CUBLAS(cublasDscal(cublasHandle, bias_size, &scale, d_bias_grads, 1));

    // Update parameters: weights -= weight_grads, biases -= bias_grads
    double alpha = -1.0; // Negative for subtraction
    CHECK_CUBLAS(cublasDaxpy(cublasHandle, m * n, &alpha, d_weight_grads, 1, d_weights, 1));
    CHECK_CUBLAS(cublasDaxpy(cublasHandle, bias_size, &alpha, d_bias_grads, 1, d_biases, 1));

    // Transfer updated parameters back to host
    CHECK_CUDA(cudaMemcpy(weights.data(), d_weights, m * n * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(biases.data(), d_biases, bias_size * sizeof(double), cudaMemcpyDeviceToHost));

    // Free device memory
    CHECK_CUDA(cudaFree(d_weights));
    CHECK_CUDA(cudaFree(d_biases));
    CHECK_CUDA(cudaFree(d_weight_grads));
    CHECK_CUDA(cudaFree(d_bias_grads));
}
