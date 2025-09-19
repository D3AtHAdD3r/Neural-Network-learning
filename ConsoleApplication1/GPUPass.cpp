#ifdef __INTELLISENSE__
#define CUDA_KERNEL_NODE_PARAMS
#define __CUDACC__
#endif

#include"GPUPass.hpp"
#include <algorithm> 
#include <iomanip>
#include"cuda_kernels.h"


double GPUPass::compute_mse_lossGPU(const Eigen::VectorXd& output, const Eigen::VectorXd& target) {
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


double GPUPass::compute_cross_entropy_lossGPU(const Eigen::VectorXd& output, const Eigen::VectorXd& target) {
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

void GPUPass::updateParametersGPU(double* d_weights,
    double* d_biases,
    const double* d_weight_grads,
    const double* d_bias_grads,
    double* d_temp_weight_grads, double* d_temp_bias_grads,
    int m, int n, int bias_size, double scale) {

    // Copy gradients to temporary buffers
    CHECK_CUBLAS(cublasDcopy(cublasHandle, m * n, d_weight_grads, 1, d_temp_weight_grads, 1));
    CHECK_CUBLAS(cublasDcopy(cublasHandle, bias_size, d_bias_grads, 1, d_temp_bias_grads, 1));

    // Scale temporary gradients: temp_grads *= scale
    CHECK_CUBLAS(cublasDscal(cublasHandle, m * n, &scale, d_temp_weight_grads, 1));
    CHECK_CUBLAS(cublasDscal(cublasHandle, bias_size, &scale, d_temp_bias_grads, 1));

    // Update parameters: weights -= temp_weight_grads, biases -= temp_bias_grads
    double alpha = -1.0;
    CHECK_CUBLAS(cublasDaxpy(cublasHandle, m * n, &alpha, d_temp_weight_grads, 1, d_weights, 1));
    CHECK_CUBLAS(cublasDaxpy(cublasHandle, bias_size, &alpha, d_temp_bias_grads, 1, d_biases, 1));

}

double GPUPass::compute_gradient_norm_gpu(
    const std::vector<double*>& weight_grads, const std::vector<double*>& bias_grads,
    const std::vector<int>& w_rows, const std::vector<int>& w_cols, const std::vector<int>& b_sizes, size_t batch_size) {
    double total_sq_norm = 0.0;
    double temp_norm = 0.0;
    for (size_t i = 0; i < weight_grads.size(); ++i) {
        //temp_norm = 0.0;
        CHECK_CUBLAS(cublasDnrm2(cublasHandle, w_rows[i] * w_cols[i], weight_grads[i], 1, &temp_norm));
        total_sq_norm += temp_norm * temp_norm;
        CHECK_CUBLAS(cublasDnrm2(cublasHandle, b_sizes[i], bias_grads[i], 1, &temp_norm));
        total_sq_norm += temp_norm * temp_norm;
    }
    return std::sqrt(total_sq_norm) / batch_size;  // Average norm per example
}

// New: Add regularization (weight_grad += scale * weights)
void GPUPass::add_regularization(double* d_weight_grad, double* d_weights, double scale, int m, int n) {
    double alpha = scale;
    cublasDaxpy(cublasHandle, m * n, &alpha, d_weights, 1, d_weight_grad, 1);
}

double GPUPass::compute_gradient_norm_gpu(
    const std::vector<double*>& weight_grads, const std::vector<double*>& bias_grads,
    const std::vector<int>& w_rows, const std::vector<int>& w_cols, const std::vector<int>& b_sizes, size_t batch_size) {
    double total_sq_norm = 0.0;
    double temp_norm = 0.0;
    for (size_t i = 0; i < weight_grads.size(); ++i) {
        //temp_norm = 0.0;
        CHECK_CUBLAS(cublasDnrm2(cublasHandle, w_rows[i] * w_cols[i], weight_grads[i], 1, &temp_norm));
        total_sq_norm += temp_norm * temp_norm;
        CHECK_CUBLAS(cublasDnrm2(cublasHandle, b_sizes[i], bias_grads[i], 1, &temp_norm));
        total_sq_norm += temp_norm * temp_norm;
    }
    return std::sqrt(total_sq_norm) / batch_size;  // Average norm per example
}

void GPUPass::computeLinearGPU_batch(const double* d_weights, const double* d_batch_input, const double* d_biases,
    double* d_batch_z, int m, int n, int batch_size) {
    // Beginner note: Compute z = W * X + b for a batch.
    // - W: weights (m × n, m=num_neurons, n=num_inputs)
    // - X: batch input (n × batch_size, each column is one input)
    // - b: biases (m, broadcast to each batch column)
    // - z: output (m × batch_size)
    // 
    // Step 1: Matrix multiply W * X using cuBLAS (result in d_batch_z)
    // Step 2: Add biases to each column using a custom kernel

    double alpha = 1.0, beta = 0.0;
    // cuBLAS uses column-major: C = alpha * A * B + beta * C
    // Here: d_batch_z = W * d_batch_input
    // - A = W (m × n), B = d_batch_input (n × batch_size), C = d_batch_z (m × batch_size)
    CHECK_CUBLAS(cublasDgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
        m, batch_size, n, &alpha,
        d_weights, m, d_batch_input, n,
        &beta, d_batch_z, m));

    // Add biases: z[:, j] += b for each batch index j
    dim3 threadsPerBlock(16, 16); // 2D grid for rows and batch
    dim3 blocksPerGrid((m + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);
    add_bias_batch << <blocksPerGrid, threadsPerBlock >> > (d_batch_z, d_biases, m, batch_size);
    CHECK_CUDA(cudaGetLastError());
}

void GPUPass::applyActivationGPU_batch(const double* d_batch_z, double* d_batch_a, int vec_size, int batch_size,
    const Activation* activation) {

    // Beginner note: Apply activation (e.g., sigmoid) to a batch matrix (vec_size × batch_size).
    // We use cuDNN, which expects 4D tensors (NCHW format: batch, channels, height, width).
    // Here: N=batch_size, C=1, H=vec_size, W=1 (treat each column as a 1D vector).
    cudnnActivationMode_t mode = activation->getCudnnActivationMode();
    cudnnTensorDescriptor_t tensorDesc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&tensorDesc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(tensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE,
        batch_size, 1, vec_size, 1));

    cudnnActivationDescriptor_t activationDesc;
    CHECK_CUDNN(cudnnCreateActivationDescriptor(&activationDesc));
    CHECK_CUDNN(cudnnSetActivationDescriptor(activationDesc, mode, CUDNN_NOT_PROPAGATE_NAN, 0.0));

    double alpha = 1.0, beta = 0.0;
    CHECK_CUDNN(cudnnActivationForward(cudnnHandle, activationDesc, &alpha, tensorDesc, d_batch_z,
        &beta, tensorDesc, d_batch_a));

    CHECK_CUDNN(cudnnDestroyTensorDescriptor(tensorDesc));
    CHECK_CUDNN(cudnnDestroyActivationDescriptor(activationDesc));
}

void GPUPass::launch_elementwise_subtract_batch(const double* a, const double* b, double* c, int rows, int batch_size) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((rows + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);
    elementwise_subtract_batch_kernel << <blocksPerGrid, threadsPerBlock >> > (a, b, c, rows, batch_size);
    CHECK_CUDA(cudaGetLastError());
}

void GPUPass::launch_elementwise_multiply_batch(const double* a, const double* b, double* c, int rows, int batch_size) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((rows + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);
    elementwise_multiply_batch_kernel << <blocksPerGrid, threadsPerBlock >> > (a, b, c, rows, batch_size);
    CHECK_CUDA(cudaGetLastError());
}

void GPUPass::computeGradientsGPU_batch(
    const double* d_deltas_batch,          // (m x batch_size)
    const double* d_prev_activations_batch,// (n x batch_size)
    double* d_weight_grads,                // (m x n)
    double* d_bias_grads,                  // (m)
    int m, int n, int batch_size)
{
    double alpha = 1.0, beta = 1.0;

    // ---- Bias grads: bias_grad += deltas * ones ----
    // Create a vector of ones on device (batch_size x 1)
    double* d_ones_batch;
    CHECK_CUDA(cudaMalloc(&d_ones_batch, batch_size * sizeof(double)));
    std::vector<double> ones(batch_size, 1.0);
    CHECK_CUDA(cudaMemcpy(d_ones_batch, ones.data(),
        batch_size * sizeof(double),
        cudaMemcpyHostToDevice));

    // GEMV: (m x batch_size) * (batch_size x 1) -> (m)
    CHECK_CUBLAS(cublasDgemv(cublasHandle, CUBLAS_OP_N, m, batch_size,
        &alpha, d_deltas_batch, m, d_ones_batch, 1,
        &beta, d_bias_grads, 1));

    CHECK_CUDA(cudaFree(d_ones_batch));

    // ---- Weight grads: weight_grad += deltas * prev_activations^T ----
    // GEMM: (m x batch_size) * (n x batch_size)^T -> (m x n)
    CHECK_CUBLAS(cublasDgemm(
        cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
        m, n, batch_size, &alpha,
        d_deltas_batch, m,
        d_prev_activations_batch, n,
        &beta, d_weight_grads, m));
}

// Batched delta propagation: delta_batch = W^T * next_delta_batch
void GPUPass::compute_delta_back_batch(const double* d_weights, const double* d_delta_next_batch, double* d_delta_batch, int m, int n, int batch_size) {
    double alpha = 1.0, beta = 0.0;
    // W^T (n x m) * next_delta (m x batch_size) -> n x batch_size
    cublasDgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, n, batch_size, m, &alpha, d_weights, m, d_delta_next_batch, m, &beta, d_delta_batch, n);
}

void GPUPass::computeActivationDerivativeGPU_batch(const double* d_pre_activations, double* d_derivatives, int vec_size, int batch_size, const Activation* activation) {
    cudnnActivationMode_t mode = activation->getCudnnActivationMode();
    if (mode == CUDNN_ACTIVATION_SIGMOID) {
        dim3 threads(16, 16);
        dim3 blocks((vec_size + threads.x - 1) / threads.x, (batch_size + threads.y - 1) / threads.y);
        sigmoid_prime_batch_kernel << <blocks, threads >> > (d_pre_activations, d_derivatives, vec_size, batch_size);
        CHECK_CUDA(cudaGetLastError());
    }
    else {
        throw std::runtime_error("Unsupported activation for batched derivative");
    }
}

// Wrapper for cost derivative (MSE or CE with sigmoid: output - target)
void GPUPass::cost_prime_mse_crossent_batched(const double* d_output, const double* d_target, double* d_delta, int rows, int batch_size) {
    launch_elementwise_subtract_batch(d_output, d_target, d_delta, rows, batch_size);
}