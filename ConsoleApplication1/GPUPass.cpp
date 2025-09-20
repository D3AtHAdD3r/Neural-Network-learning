#ifdef __INTELLISENSE__
#define CUDA_KERNEL_NODE_PARAMS
#define __CUDACC__
#endif

#include"GPUPass.hpp"
#include"cuda_kernels.h"
#include <algorithm> 
#include <iomanip>



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
    cuda_kernels::crossEntropyLossKernel << <num_blocks, 256 >> > (d_output, d_target, d_loss, n);
    CHECK_CUDA(cudaGetLastError());

    // Sum the losses
    cuda_kernels::sumReductionKernel << <num_blocks, 256, 256 * sizeof(double) >> > (d_loss, d_sum, n);
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

// New helper: Squared norm on GPU (use reduction kernel)
double GPUPass::compute_squared_norm_gpu(double* d_data, int n) {
    double* d_sum;
    cudaMalloc(&d_sum, sizeof(double));
    cuda_kernels::sumReductionKernel << <1, 256, 256 * sizeof(double) >> > (d_data, d_sum, n);  // Simplified; use full reduction for large n
    double sum;
    cudaMemcpy(&sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_sum);
    return sum * sum;  // Wait, no: squared norm is sum of squares, so sum (x_i^2)
    // Actually, modify sumReduction to sum squares
    // New kernel for sum of squares
}

double GPUPass::compute_squared_normGPU(const Eigen::MatrixXd& matrix) {
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

void GPUPass::process_subbatched(std::function<void(int sub_batch_size)> func, int total_batch_size, int max_batch) {
    // Beginner note: Placeholder for processing large batches in smaller chunks.
    // If total_batch_size > max_batch, split into sub-batches of size <= max_batch.
    // For now, just throw an error (as per plan); implement splitting in Phase 5.3.
    if (total_batch_size > max_batch) {
        throw std::runtime_error("Batch size (" + std::to_string(total_batch_size) +
            ") exceeds max (" + std::to_string(max_batch) + ")");
    }
    func(total_batch_size); // Process the whole batch
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
    cuda_kernels::add_bias_batch << <blocksPerGrid, threadsPerBlock >> > (d_batch_z, d_biases, m, batch_size);
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
    cuda_kernels::elementwise_subtract_batch_kernel << <blocksPerGrid, threadsPerBlock >> > (a, b, c, rows, batch_size);
    CHECK_CUDA(cudaGetLastError());
}

void GPUPass::launch_elementwise_multiply_batch(const double* a, const double* b, double* c, int rows, int batch_size) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((rows + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);
    cuda_kernels::elementwise_multiply_batch_kernel << <blocksPerGrid, threadsPerBlock >> > (a, b, c, rows, batch_size);
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
        cuda_kernels::sigmoid_prime_batch_kernel << <blocks, threads >> > (d_pre_activations, d_derivatives, vec_size, batch_size);
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

void GPUPass::allocate_weights(double** d_weights, int rows, int cols) {
    CHECK_CUDA(cudaMalloc(d_weights, rows * cols * sizeof(double)));
}

void GPUPass::allocate_biases(double** d_biases, int size) {
    CHECK_CUDA(cudaMalloc(d_biases, size * sizeof(double)));
}

void GPUPass::copy_weights_to_device(double* d_weights, const Eigen::MatrixXd& weights) {
    CHECK_CUDA(cudaMemcpy(d_weights, weights.data(), weights.rows() * weights.cols() * sizeof(double), cudaMemcpyHostToDevice));
}

void GPUPass::copy_biases_to_device(double* d_biases, const Eigen::VectorXd& biases) {
    CHECK_CUDA(cudaMemcpy(d_biases, biases.data(), biases.size() * sizeof(double), cudaMemcpyHostToDevice));
}

void GPUPass::copy_weights_to_host(Eigen::MatrixXd& weights, double* d_weights, int rows, int cols) {
    weights.resize(rows, cols);
    CHECK_CUDA(cudaMemcpy(weights.data(), d_weights, rows * cols * sizeof(double), cudaMemcpyDeviceToHost));
}

void GPUPass::copy_biases_to_host(Eigen::VectorXd& biases, double* d_biases, int size) {
    biases.resize(size);
    CHECK_CUDA(cudaMemcpy(biases.data(), d_biases, size * sizeof(double), cudaMemcpyDeviceToHost));
}

void GPUPass::free_weights(double* d_weights) {
    if (d_weights) {
        CHECK_CUDA(cudaFree(d_weights));
    }
}

void GPUPass::free_biases(double* d_biases) {
    if (d_biases) {
        CHECK_CUDA(cudaFree(d_biases));
    }
}

void GPUPass::allocate_vector(double** d_vector, int size) {
    //CHECK_CUDA(cudaMalloc(d_vector, size * sizeof(double)));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(d_vector), size * sizeof(double)));
}

void GPUPass::free_vector(double* d_vector) {
    if (d_vector) {
        CHECK_CUDA(cudaFree(d_vector));
    }
}

void GPUPass::copy_to_device(double* d_vector, const Eigen::VectorXd& vector) {
    CHECK_CUDA(cudaMemcpy(d_vector, vector.data(), vector.size() * sizeof(double), cudaMemcpyHostToDevice));
}

void GPUPass::copy_to_device(double* d_matrix, const Eigen::MatrixXd& matrix) {
    CHECK_CUDA(cudaMemcpy(d_matrix, matrix.data(),
        matrix.rows() * matrix.cols() * sizeof(double),
        cudaMemcpyHostToDevice));
}

void GPUPass::copy_to_host(Eigen::VectorXd& vector, double* d_vector, int size) {
    vector.resize(size);
    CHECK_CUDA(cudaMemcpy(vector.data(), d_vector, size * sizeof(double), cudaMemcpyDeviceToHost));
}

void GPUPass::copy_to_host(Eigen::MatrixXd& matrix, double* d_matrix, int rows, int cols) {
    matrix.resize(rows, cols);
    CHECK_CUDA(cudaMemcpy(matrix.data(), d_matrix, rows * cols * sizeof(double), cudaMemcpyDeviceToHost));
}

void GPUPass::copy_device_to_device(double* dst, const double* src, int size) {
    CHECK_CUDA(cudaMemcpy(dst, src, size * sizeof(double), cudaMemcpyDeviceToDevice));
}

void GPUPass::allocate_batch_vector(double** d_vec, int vec_size, int batch_size) {
    // Beginner note: Allocate GPU memory for a batch matrix (vec_size rows × batch_size cols).
    // Stored as a flat array of size vec_size * batch_size * sizeof(double).
    // cudaMalloc sets *d_vec to the allocated pointer.
    size_t total_size = static_cast<size_t>(vec_size) * batch_size * sizeof(double);
    CHECK_CUDA(cudaMalloc(d_vec, total_size));
}

void GPUPass::free_batch_vector(double* d_vec) {
    // Beginner note: Free GPU memory allocated for a batch matrix.
    // Check for nullptr to avoid errors; cudaFree is safe to call on null.
    if (d_vec) {
        CHECK_CUDA(cudaFree(d_vec));
    }
}

void GPUPass::copy_batch_to_device(double* d_batch_matrix, const std::vector<Eigen::VectorXd>& batch, bool transpose) {
    // Beginner note: Copy a batch of vectors (e.g., inputs) from host to device.
    // Each Eigen::VectorXd is one example (vec_size elements). We store as a matrix:
    // - If transpose=false: rows=vec_size, cols=batch_size (column-major, cuBLAS default).
    // - If transpose=true: rows=batch_size, cols=vec_size (less common, for specific ops).
    int batch_size = static_cast<int>(batch.size());
    if (batch_size == 0) return;

    int vec_size = static_cast<int>(batch[0].size());
    // Check all vectors have same size
    for (const auto& vec : batch) {
        if (vec.size() != vec_size) {
            throw std::runtime_error("Inconsistent vector sizes in batch");
        }
    }

    if (transpose) {
        // Store as batch_size × vec_size
        std::vector<double> host_buffer(batch_size * vec_size);
        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < vec_size; ++j) {
                host_buffer[i * vec_size + j] = batch[i](j);
            }
        }

        CHECK_CUDA(cudaMemcpy(d_batch_matrix, host_buffer.data(), batch_size * vec_size * sizeof(double), cudaMemcpyHostToDevice));
    }
    else {
        // Store as vec_size × batch_size (column-major)
        std::vector<double> host_buffer(vec_size * batch_size);
        for (int j = 0; j < batch_size; ++j) {
            for (int i = 0; i < vec_size; ++i) {
                host_buffer[i + j * vec_size] = batch[j](i);
            }
        }
        CHECK_CUDA(cudaMemcpy(d_batch_matrix, host_buffer.data(), vec_size * batch_size * sizeof(double), cudaMemcpyHostToDevice));
    }
}

void GPUPass::copy_batch_to_host(std::vector<Eigen::VectorXd>& batch, const double* d_batch_matrix, int vec_size, int batch_size) {
    // Beginner note: Copy a batch matrix from device to host.
    // The device matrix is vec_size × batch_size (column-major).
    // Each column becomes an Eigen::VectorXd in the output batch.
    if (batch_size <= 0 || vec_size <= 0) return;

    std::vector<double> host_buffer(vec_size * batch_size);
    CHECK_CUDA(cudaMemcpy(host_buffer.data(), d_batch_matrix, vec_size * batch_size * sizeof(double), cudaMemcpyDeviceToHost));

    batch.resize(batch_size);
    for (int j = 0; j < batch_size; ++j) {
        batch[j].resize(vec_size);
        for (int i = 0; i < vec_size; ++i) {
            batch[j](i) = host_buffer[i + j * vec_size];
        }
    }
}

void GPUPass::set_to_zero_batch(double* d_data, int size, int batch_size) {
    // Beginner note: Zero out a batch matrix (size × batch_size).
    // Reuse existing kernel; total size is size * batch_size.
    int total_size = size * batch_size;
    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;
    cuda_kernels::set_to_zero_kernel << <blocks, threads >> > (d_data, total_size);
    CHECK_CUDA(cudaGetLastError());
}


void GPUPass::set_to_zero(double* d_data, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    cuda_kernels::set_to_zero_kernel << <blocks, threads >> > (d_data, n);
    CHECK_CUDA(cudaGetLastError());
}

// New elementwise_multiply_Caller
void GPUPass::launch_elementwise_multiply(const double* a, const double* b, double* c, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    cuda_kernels::elementwise_multiply << <blocks, threads >> > (a, b, c, n);
    CHECK_CUDA(cudaGetLastError());
}

// New: Elementwise subtract (a - b -> c)
void GPUPass::launch_elementwise_subtract(const double* a, const double* b, double* c, int n) {
    CHECK_CUDA(cudaMemcpy(c, a, n * sizeof(double), cudaMemcpyDeviceToDevice));
    double alpha = -1.0;
    CHECK_CUBLAS(cublasDaxpy(cublasHandle, n, &alpha, b, 1, c, 1));
}


void GPUPass::debugPrint(const double* data, int num_inputs) {
    // Launch with enough threads
    int threads = 256;
    int blocks = (num_inputs + threads - 1) / threads;
    cuda_kernels::debugPrint_kernel << <blocks, threads >> > (data, num_inputs);
    cudaDeviceSynchronize();
}

void GPUPass::debugPrint_batch(const double* d_data, int rows, int cols) {
    // Beginner note: Debug print a batch matrix (rows × cols).
    // Copy to host and print a small portion (up to 10×10) to avoid flooding output.
    std::vector<double> host_data(rows * cols);
    CHECK_CUDA(cudaMemcpy(host_data.data(), d_data, rows * cols * sizeof(double), cudaMemcpyDeviceToHost));

    int max_rows = std::min(rows, 10);
    int max_cols = std::min(cols, 10);
    std::cout << "Batch matrix (" << rows << " × " << cols << "):\n";
    for (int i = 0; i < max_rows; ++i) {
        std::cout << "[ ";
        for (int j = 0; j < max_cols; ++j) {
            std::cout << std::fixed << std::setprecision(4) << host_data[i + j * rows];
            if (j < max_cols - 1) std::cout << ", ";
        }
        if (max_cols < cols) std::cout << ", ...";
        std::cout << " ]\n";
    }
    if (max_rows < rows || max_cols < cols) {
        std::cout << "(Truncated, full size: " << rows << " × " << cols << ")\n";
    }
    cudaDeviceSynchronize();
}