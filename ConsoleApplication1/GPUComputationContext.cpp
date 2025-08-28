#ifdef __INTELLISENSE__
#define CUDA_KERNEL_NODE_PARAMS
#define __CUDACC__
#endif

#include"GPUComputationContext.hpp"

// CUDA kernel for element-wise multiplication
__global__ void elementwise_multiply(const double* a, const double* b, double* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

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

//debug kernel
__global__ void debugPrint_kernel(const double* data, int n) {
    /*int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        printf("data[%d] = %f\n", idx, data[idx]);
    }*/

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) printf("data[%d] = %f\n", idx, data[idx]);

}

// New kernel for setting array to zero
__global__ void set_to_zero_kernel(double* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = 0.0;
    }
}

// New kernel for sum of squares
__global__ void sum_squares_reduction(const double* input, double* output, int n) {
    // Similar to sumReductionKernel, but sdata[tid] = (idx < n) ? input[idx] * input[idx] : 0.0;
}

// New: Elementwise subtract (a - b -> c)
void GPUComputationContext::launch_elementwise_subtract(const double* a, const double* b, double* c, int n) {
    CHECK_CUDA(cudaMemcpy(c, a, n * sizeof(double), cudaMemcpyDeviceToDevice));
    double alpha = -1.0;
    CHECK_CUBLAS(cublasDaxpy(cublasHandle, n, &alpha, b, 1, c, 1));
}

// New method
void GPUComputationContext::set_to_zero(double* d_data, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    set_to_zero_kernel << <blocks, threads >> > (d_data, n);
    CHECK_CUDA(cudaGetLastError());
}

// New: Add regularization (weight_grad += scale * weights)
void GPUComputationContext::add_regularization(double* d_weight_grad, double* d_weights, double scale, int m, int n) {
    double alpha = scale;
    cublasDaxpy(cublasHandle, m * n, &alpha, d_weights, 1, d_weight_grad, 1);
}

// New: Compute delta back (delta = W^T * delta_next)
void GPUComputationContext::compute_delta_back(double* d_weights, double* d_delta_next, double* d_delta, int m, int n) {
    double alpha = 1.0, beta = 0.0;
    cublasDgemv(cublasHandle, CUBLAS_OP_T, m, n, &alpha, d_weights, m, d_delta_next, 1, &beta, d_delta, 1);
}

// New: Compute total gradient norm across all layers/batches
//double GPUComputationContext::compute_gradient_norm_gpu(
//    const std::vector<double*>& weight_grads, const std::vector<double*>& bias_grads,
//    const std::vector<int>& w_rows, const std::vector<int>& w_cols, const std::vector<int>& b_sizes, size_t batch_size) {
//    double total_sq_norm = 0.0;
//    for (size_t i = 0; i < weight_grads.size(); ++i) {
//        total_sq_norm += compute_squared_norm_gpu(weight_grads[i], w_rows[i] * w_cols[i]);
//        total_sq_norm += compute_squared_norm_gpu(bias_grads[i], b_sizes[i]);
//    }
//    return std::sqrt(total_sq_norm / batch_size);
//}

double GPUComputationContext::compute_gradient_norm_gpu(
    const std::vector<double*>& weight_grads, const std::vector<double*>& bias_grads,
    const std::vector<int>& w_rows, const std::vector<int>& w_cols, const std::vector<int>& b_sizes, size_t batch_size) {
    double total_sq_norm = 0.0;
    double temp_norm;
    for (size_t i = 0; i < weight_grads.size(); ++i) {
        CHECK_CUBLAS(cublasDnrm2(cublasHandle, w_rows[i] * w_cols[i], weight_grads[i], 1, &temp_norm));
        total_sq_norm += temp_norm * temp_norm;
        CHECK_CUBLAS(cublasDnrm2(cublasHandle, b_sizes[i], bias_grads[i], 1, &temp_norm));
        total_sq_norm += temp_norm * temp_norm;
    }
    return std::sqrt(total_sq_norm) / batch_size;  // Average norm per example
}

// New helper: Squared norm on GPU (use reduction kernel)
double GPUComputationContext::compute_squared_norm_gpu(double* d_data, int n) {
    double* d_sum;
    cudaMalloc(&d_sum, sizeof(double));
    sumReductionKernel << <1, 256, 256 * sizeof(double) >> > (d_data, d_sum, n);  // Simplified; use full reduction for large n
    double sum;
    cudaMemcpy(&sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_sum);
    return sum * sum;  // Wait, no: squared norm is sum of squares, so sum (x_i^2)
    // Actually, modify sumReduction to sum squares
    // New kernel for sum of squares
}

// New elementwise_multiply_Caller
void GPUComputationContext::launch_elementwise_multiply(const double* a, const double* b, double* c, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    elementwise_multiply << <blocks, threads >> > (a, b, c, n);
    CHECK_CUDA(cudaGetLastError());
}



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

Eigen::MatrixXd GPUComputationContext::computeWeightGradient(const Eigen::VectorXd& delta, const Eigen::VectorXd& activation)
{
    int m = delta.size();
    int n = activation.size();
    Eigen::MatrixXd result(m, n);

    double* d_delta, * d_activation, * d_result;
    CHECK_CUDA(cudaMalloc(&d_delta, m * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_activation, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_result, m * n * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(d_delta, delta.data(), m * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_activation, activation.data(), n * sizeof(double), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemset(d_result, 0, m * n * sizeof(double))); // Ensure clean slate
    double alpha = 1.0;
    CHECK_CUBLAS(cublasDger(cublasHandle, m, n, &alpha, d_delta, 1, d_activation, 1, d_result, m));

    CHECK_CUDA(cudaMemcpy(result.data(), d_result, m * n * sizeof(double), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_delta));
    CHECK_CUDA(cudaFree(d_activation));
    CHECK_CUDA(cudaFree(d_result));

    return result;
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

void GPUComputationContext::accumulateGradients(const std::vector<Eigen::MatrixXd>& weight_grads_in, const std::vector<Eigen::VectorXd>& bias_grads_in, std::vector<Eigen::MatrixXd>& weight_grads_out, std::vector<Eigen::VectorXd>& bias_grads_out, double scale)
{
    if (weight_grads_in.size() != weight_grads_out.size() || bias_grads_in.size() != bias_grads_out.size()) {
        throw std::runtime_error("Gradient vector size mismatch in accumulateGradients");
    }

    for (size_t i = 0; i < weight_grads_in.size(); ++i) {
        int m = weight_grads_in[i].rows();
        int n = weight_grads_in[i].cols();
        int bias_size = bias_grads_in[i].size();

        // Initialize output gradients if empty
        if (weight_grads_out[i].size() == 0) {
            weight_grads_out[i].resize(m, n);
            weight_grads_out[i].setZero();
        }
        if (bias_grads_out[i].size() == 0) {
            bias_grads_out[i].resize(bias_size);
            bias_grads_out[i].setZero();
        }

        // Device memory pointers
        double* d_weight_grads_in, * d_weight_grads_out, * d_bias_grads_in, * d_bias_grads_out;

        // Allocate device memory
        CHECK_CUDA(cudaMalloc(&d_weight_grads_in, m * n * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_weight_grads_out, m * n * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_bias_grads_in, bias_size * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_bias_grads_out, bias_size * sizeof(double)));

        // Copy input and output gradients to device
        CHECK_CUDA(cudaMemcpy(d_weight_grads_in, weight_grads_in[i].data(), m * n * sizeof(double), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_weight_grads_out, weight_grads_out[i].data(), m * n * sizeof(double), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_bias_grads_in, bias_grads_in[i].data(), bias_size * sizeof(double), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_bias_grads_out, bias_grads_out[i].data(), bias_size * sizeof(double), cudaMemcpyHostToDevice));

        // Accumulate: out += scale * in
        double alpha = scale;
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, m * n, &alpha, d_weight_grads_in, 1, d_weight_grads_out, 1));
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, bias_size, &alpha, d_bias_grads_in, 1, d_bias_grads_out, 1));

        // Copy back to host
        CHECK_CUDA(cudaMemcpy(weight_grads_out[i].data(), d_weight_grads_out, m * n * sizeof(double), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(bias_grads_out[i].data(), d_bias_grads_out, bias_size * sizeof(double), cudaMemcpyDeviceToHost));

        // Free device memory
        CHECK_CUDA(cudaFree(d_weight_grads_in));
        CHECK_CUDA(cudaFree(d_weight_grads_out));
        CHECK_CUDA(cudaFree(d_bias_grads_in));
        CHECK_CUDA(cudaFree(d_bias_grads_out));

    }
}

double GPUComputationContext::compute_squared_norm(const Eigen::MatrixXd& matrix) {
    int m = matrix.rows();
    int n = matrix.cols();
    double* d_matrix;
    CHECK_CUDA(cudaMalloc(&d_matrix, m * n * sizeof(double)));
    CHECK_CUDA(cudaMemcpy(d_matrix, matrix.data(), m * n * sizeof(double), cudaMemcpyHostToDevice));

    double norm;
    CHECK_CUBLAS(cublasDnrm2(cublasHandle, m * n, d_matrix, 1, &norm));

    CHECK_CUDA(cudaFree(d_matrix));
    return norm * norm;
}

double GPUComputationContext::compute_mse_loss(const Eigen::VectorXd& output, const Eigen::VectorXd& target) {
    int n = output.size();
    if (n != target.size()) {
        throw std::runtime_error("Mismatched sizes in compute_mse_loss");
    }

    double* d_output, * d_target, * d_diff;
    CHECK_CUDA(cudaMalloc(&d_output, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_target, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_diff, n * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(d_output, output.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_target, target.data(), n * sizeof(double), cudaMemcpyHostToDevice));

    // Compute diff = output - target
    double alpha = -1.0;
    CHECK_CUDA(cudaMemcpy(d_diff, d_output, n * sizeof(double), cudaMemcpyDeviceToDevice));
    CHECK_CUBLAS(cublasDaxpy(cublasHandle, n, &alpha, d_target, 1, d_diff, 1));

    // Compute squared norm
    double norm;
    CHECK_CUBLAS(cublasDnrm2(cublasHandle, n, d_diff, 1, &norm));

    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_target));
    CHECK_CUDA(cudaFree(d_diff));

    return norm * norm;
}

double GPUComputationContext::compute_cross_entropy_loss(const Eigen::VectorXd& output, const Eigen::VectorXd& target) {
    int n = output.size();
    if (n != target.size()) {
        throw std::runtime_error("Mismatched sizes in compute_cross_entropy_loss");
    }

    double* d_output, * d_target, * d_loss, * d_sum;
    CHECK_CUDA(cudaMalloc(&d_output, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_target, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_loss, n * sizeof(double)));
    int num_blocks = (n + 255) / 256;
    CHECK_CUDA(cudaMalloc(&d_sum, num_blocks * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(d_output, output.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_target, target.data(), n * sizeof(double), cudaMemcpyHostToDevice));

    // Compute element-wise cross-entropy loss
    crossEntropyLossKernel << <num_blocks, 256 >> > (d_output, d_target, d_loss, n);
    CHECK_CUDA(cudaGetLastError());

    // Sum the losses
    sumReductionKernel << <num_blocks, 256, 256 * sizeof(double) >> > (d_loss, d_sum, n);
    CHECK_CUDA(cudaGetLastError());

    // Copy partial sums to host and complete reduction
    std::vector<double> partial_sums(num_blocks);
    CHECK_CUDA(cudaMemcpy(partial_sums.data(), d_sum, num_blocks * sizeof(double), cudaMemcpyDeviceToHost));

    double total_loss = 0.0;
    for (double sum : partial_sums) {
        total_loss += sum;
    }

    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_target));
    CHECK_CUDA(cudaFree(d_loss));
    CHECK_CUDA(cudaFree(d_sum));

    return total_loss;
}

//-----new funcs after gpu compute optimization-------//

void GPUComputationContext::allocate_weights(double** d_weights, int rows, int cols) {
    CHECK_CUDA(cudaMalloc(d_weights, rows * cols * sizeof(double)));
}

void GPUComputationContext::allocate_biases(double** d_biases, int size) {
    CHECK_CUDA(cudaMalloc(d_biases, size * sizeof(double)));
}

void GPUComputationContext::copy_weights_to_device(double* d_weights, const Eigen::MatrixXd& weights) {
    CHECK_CUDA(cudaMemcpy(d_weights, weights.data(), weights.rows() * weights.cols() * sizeof(double), cudaMemcpyHostToDevice));
}

void GPUComputationContext::copy_biases_to_device(double* d_biases, const Eigen::VectorXd& biases) {
    CHECK_CUDA(cudaMemcpy(d_biases, biases.data(), biases.size() * sizeof(double), cudaMemcpyHostToDevice));
}

void GPUComputationContext::copy_weights_to_host(Eigen::MatrixXd& weights, double* d_weights, int rows, int cols) {
    weights.resize(rows, cols);
    CHECK_CUDA(cudaMemcpy(weights.data(), d_weights, rows * cols * sizeof(double), cudaMemcpyDeviceToHost));
}

void GPUComputationContext::copy_biases_to_host(Eigen::VectorXd& biases, double* d_biases, int size) {
    biases.resize(size);
    CHECK_CUDA(cudaMemcpy(biases.data(), d_biases, size * sizeof(double), cudaMemcpyDeviceToHost));
}

void GPUComputationContext::free_weights(double* d_weights) {
    if (d_weights) {
        CHECK_CUDA(cudaFree(d_weights));
    }
}

void GPUComputationContext::free_biases(double* d_biases) {
    if (d_biases) {
        CHECK_CUDA(cudaFree(d_biases));
    }
}

void GPUComputationContext::allocate_vector(double** d_vector, int size) {
    //CHECK_CUDA(cudaMalloc(d_vector, size * sizeof(double)));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(d_vector), size * sizeof(double)));
}

void GPUComputationContext::free_vector(double* d_vector) {
    if (d_vector) {
        CHECK_CUDA(cudaFree(d_vector));
    }
}

void GPUComputationContext::copy_to_device(double* d_vector, const Eigen::VectorXd& vector) {
    CHECK_CUDA(cudaMemcpy(d_vector, vector.data(), vector.size() * sizeof(double), cudaMemcpyHostToDevice));
}

void GPUComputationContext::copy_to_device(double* d_matrix, const Eigen::MatrixXd& matrix) {
    CHECK_CUDA(cudaMemcpy(d_matrix, matrix.data(),
        matrix.rows() * matrix.cols() * sizeof(double),
        cudaMemcpyHostToDevice));
}

void GPUComputationContext::copy_to_host(Eigen::VectorXd& vector, double* d_vector, int size) {
    vector.resize(size);
    CHECK_CUDA(cudaMemcpy(vector.data(), d_vector, size * sizeof(double), cudaMemcpyDeviceToHost));
}

void GPUComputationContext::copy_to_host(Eigen::MatrixXd& matrix, double* d_matrix, int rows, int cols) {
    matrix.resize(rows, cols);
    CHECK_CUDA(cudaMemcpy(matrix.data(), d_matrix, rows * cols * sizeof(double), cudaMemcpyDeviceToHost));
}


void GPUComputationContext::computeLinearGPU(double* d_weights, double* d_input, double* d_biases, double* d_z, int m, int n) {
    double alpha = 1.0, beta = 0.0;
    CHECK_CUBLAS(cublasDgemv(cublasHandle, CUBLAS_OP_N, m, n, 
        &alpha, d_weights, m, d_input, 1, &beta, d_z, 1));
    alpha = 1.0;
    CHECK_CUBLAS(cublasDaxpy(cublasHandle, m, &alpha, d_biases, 1, d_z, 1));
}


void GPUComputationContext::applyActivationGPU(double* d_z, double* d_a, int n, const Activation* activation) {
    cudnnActivationMode_t mode = activation->getCudnnActivationMode();
    cudnnTensorDescriptor_t tensorDesc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&tensorDesc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(tensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, 1, n, 1));
    cudnnActivationDescriptor_t activationDesc;
    CHECK_CUDNN(cudnnCreateActivationDescriptor(&activationDesc));
    CHECK_CUDNN(cudnnSetActivationDescriptor(activationDesc, mode, CUDNN_NOT_PROPAGATE_NAN, 0.0));
    double alpha = 1.0, beta = 0.0;
    CHECK_CUDNN(cudnnActivationForward(cudnnHandle, activationDesc, &alpha, tensorDesc, d_z, &beta, tensorDesc, d_a));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(tensorDesc));
    CHECK_CUDNN(cudnnDestroyActivationDescriptor(activationDesc));
}

Eigen::VectorXd GPUComputationContext::computeActivationDerivativeGPU(double* d_a, double* d_z, double* d_dy, double* d_derivatives, int size, const Activation* activation) {
    if (!activation) {
        throw std::runtime_error("Null activation pointer in computeActivationDerivative");
    }

    int n = size;

    // Get cuDNN activation mode
    cudnnActivationMode_t mode = activation->getCudnnActivationMode();

    //TODO: can be moved to layer and implemented once after initializing d_dy?
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
        d_a, tensorDesc, d_dy, tensorDesc, d_z,
        &beta, tensorDesc, d_derivatives));

    // Transfer result back to host
    Eigen::VectorXd derivatives(n);
    CHECK_CUDA(cudaMemcpy(derivatives.data(), d_derivatives, n * sizeof(double), cudaMemcpyDeviceToHost));

    // Clean up
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(tensorDesc));
    CHECK_CUDNN(cudnnDestroyActivationDescriptor(activationDesc));

    return derivatives;
}

void GPUComputationContext::debugPrint(const double* data, int num_inputs) {
    // Launch with enough threads
    int threads = 256;
    int blocks = (num_inputs + threads - 1) / threads;
    debugPrint_kernel << <blocks, threads >> > (data, num_inputs);
    cudaDeviceSynchronize();
}

void GPUComputationContext::computeGradientsGPU( 
    double* d_incoming_deltas,   // incoming deltas (on device)
    double* d_input,             // input vector (on device)
    double* d_derivatives,       // activation derivatives (on device, computed beforehand if needed)
    double* d_weight_grads,      // output: weight gradients
    double* d_bias_grads,        // output: bias gradients
    double* d_temp,              // temp buffer, same size as num_neurons
    int num_neurons,
    int num_inputs,
    bool apply_derivative
) {
    double* d_adjusted_deltas = nullptr;

    if (apply_derivative) {
        // multiply incoming_deltas * derivatives → temp
        launch_elementwise_multiply(d_incoming_deltas, d_derivatives, d_temp, num_neurons);
        d_adjusted_deltas = d_temp;
    }
    else {
        d_adjusted_deltas = d_incoming_deltas;
    }

    // reset weight grads
    set_to_zero(d_weight_grads, num_neurons * num_inputs);

    // weight grads = outer(adjusted_deltas, input)
    double alpha = 1.0;
    CHECK_CUBLAS(cublasDger(cublasHandle,
        num_neurons, num_inputs,
        &alpha,
        d_adjusted_deltas, 1,
        d_input, 1,
        d_weight_grads, num_neurons));

    // bias grads (a.k.a. delta) = adjusted_deltas
    CHECK_CUDA(cudaMemcpy(d_bias_grads, d_adjusted_deltas, num_neurons * sizeof(double), cudaMemcpyDeviceToDevice));
}


void GPUComputationContext::updateParametersGPU(double* d_weights,
    double* d_biases,
    double* d_weight_grads,
    double* d_bias_grads,
    int m, int n, int bias_size, double scale) {

    
    // Scale gradients: weight_grads *= scale, bias_grads *= scale
    CHECK_CUBLAS(cublasDscal(cublasHandle, m * n, &scale, d_weight_grads, 1));
    CHECK_CUBLAS(cublasDscal(cublasHandle, bias_size, &scale, d_bias_grads, 1));

    // Update parameters: weights -= weight_grads, biases -= bias_grads
    double alpha = -1.0;
    CHECK_CUBLAS(cublasDaxpy(cublasHandle, m * n, &alpha, d_weight_grads, 1, d_weights, 1));
    CHECK_CUBLAS(cublasDaxpy(cublasHandle, bias_size, &alpha, d_bias_grads, 1, d_biases, 1));
}


void GPUComputationContext::accumulateGradientsGPU(
    const std::vector<double*>& weight_grads_in,
    const std::vector<double*>& bias_grads_in,
    const std::vector<double*>& weight_grads_out,
    const std::vector<double*>& bias_grads_out,
    const std::vector<int>& weight_rows,
    const std::vector<int>& weight_cols,
    const std::vector<int>& bias_sizes,
    double scale) {
    // Check size consistency
    if (weight_grads_in.size() != weight_grads_out.size() ||
        bias_grads_in.size() != bias_grads_out.size() ||
        weight_grads_in.size() != weight_rows.size() ||
        weight_grads_in.size() != weight_cols.size() ||
        bias_grads_in.size() != bias_sizes.size()) {
        throw std::runtime_error("Gradient vector or size mismatch in accumulateGradientsGPU");
    }

    // Iterate over layers
    for (size_t i = 0; i < weight_grads_in.size(); ++i) {
        int m = weight_rows[i];      // Number of neurons (rows of weight matrix)
        int n = weight_cols[i];      // Number of inputs (columns of weight matrix)
        int bias_size = bias_sizes[i]; // Size of bias vector

        // Ensure pointers are valid
        if (!weight_grads_in[i] || !weight_grads_out[i] || !bias_grads_in[i] || !bias_grads_out[i]) {
            throw std::runtime_error("Null gradient pointer in accumulateGradientsGPU");
        }

        // Accumulate: out += scale * in
        double alpha = scale;
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, m * n, &alpha, weight_grads_in[i], 1, weight_grads_out[i], 1));
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, bias_size, &alpha, bias_grads_in[i], 1, bias_grads_out[i], 1));
    }
}