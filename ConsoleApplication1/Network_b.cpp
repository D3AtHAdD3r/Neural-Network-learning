#include"Network_b.hpp"
#include"SigmoidActivation.hpp"
#include <numeric>
#include <iomanip>

/**
 * @brief Constructs a network with specified layer sizes.
 * Initializes layers with Xavier-initialized weights and biases.
 * @param sizes Vector of layer sizes (e.g., {784, 30, 10} for MNIST)
 */
Network_b::Network_b(const std::vector<int>& sizes, double lambda, LossType loss_type, NeuronType neuron_type, unsigned int seed, int max_batch_size)
    :
    sizes(sizes), num_layers(sizes.size()),
    rng(seed), last_test_loss(0.0), lambda(lambda),
    loss_type_(loss_type), neuron_type_(neuron_type),
    max_batch_size_(max_batch_size),
    allocated_batch_size_(0),
    batch_buffers_allocated_(false) {

    //TODO: Error check input

    //Init GPUPass
    gpuCtx_ = std::make_unique<GPUPass>();

    if (max_batch_size_ == MAX_BATCH_SIZE) {
        int approx_net_size = std::accumulate(sizes.begin(), sizes.end(), 0);  // Rough sum of all layer sizes
        max_batch_size_ = gpuCtx_->get_dynamic_max_batch_size(approx_net_size);
        // Cap at default to be safe
        max_batch_size_ = std::min(max_batch_size_, MAX_BATCH_SIZE);
    }

    // Initialize activation
    switch (neuron_type_) {
    case NeuronType::SIGMOID:
        activation_ = std::make_unique<SigmoidActivation>();
        break;
    default:
        throw std::runtime_error("Unsupported neuron type");
    }

    // Initialize layers
    for (size_t i = 1; i < sizes.size(); ++i) {
        layers.emplace_back(std::make_unique<Layer_b>(
            sizes[i - 1], sizes[i], activation_.get(), gpuCtx_.get(), static_cast<unsigned int>(rng())
        ));
    }

    //Initialize GPU storage pointers
    
    //Initialize device memory pointers
    for (size_t i = 0; i < sizes.size() - 1; ++i) {
        double* weightGrad_currentLayer = nullptr;
        double* biasGrad_currentLayer = nullptr;
        double* temp_weightGrad = nullptr;
        double* temp_biasGrad = nullptr;

        gpuCtx_->allocate_weights(&weightGrad_currentLayer, layers[i]->get_num_neurons(), layers[i]->get_num_inputs());
        gpuCtx_->allocate_biases(&biasGrad_currentLayer, layers[i]->get_num_neurons());
        gpuCtx_->allocate_weights(&temp_weightGrad, layers[i]->get_num_neurons(), layers[i]->get_num_inputs());
        gpuCtx_->allocate_biases(&temp_biasGrad, layers[i]->get_num_neurons());

        Eigen::MatrixXd zeroMatrix = Eigen::MatrixXd::Zero(layers[i]->get_num_neurons(), layers[i]->get_num_inputs());
        Eigen::VectorXd zeroVec = Eigen::VectorXd::Zero(layers[i]->get_num_neurons());

        gpuCtx_->copy_to_device(weightGrad_currentLayer, zeroMatrix);
        gpuCtx_->copy_to_device(biasGrad_currentLayer, zeroVec);
        gpuCtx_->copy_to_device(temp_weightGrad, zeroMatrix);
        gpuCtx_->copy_to_device(temp_biasGrad, zeroVec);

        accumulate_weight_grads.push_back(weightGrad_currentLayer);
        accumulate_bias_grads.push_back(biasGrad_currentLayer);
        temp_weight_grads.push_back(temp_weightGrad);
        temp_bias_grads.push_back(temp_biasGrad);

        weight_rows.push_back(layers[i]->get_num_neurons());
        weight_cols.push_back(layers[i]->get_num_inputs());
        bias_sizes.push_back(layers[i]->get_num_neurons());
    }

    gpuCtx_->allocate_vector(&d_input_main, sizes[0]);
    gpuCtx_->set_to_zero(d_input_main, sizes[0]);
}

Network_b::~Network_b() {
    for (auto ptr : accumulate_weight_grads) {
        gpuCtx_->free_weights(ptr);
    }
    for (auto ptr : accumulate_bias_grads) {
        gpuCtx_->free_biases(ptr);
    }
    for (auto ptr : temp_weight_grads) {
        gpuCtx_->free_weights(ptr);
    }
    for (auto ptr : temp_bias_grads) {
        gpuCtx_->free_biases(ptr);
    }

    gpuCtx_->free_vector(d_input_main);

    // Free batch buffers if allocated
    if (batch_buffers_allocated_) {
        gpuCtx_->free_vector(d_batch_main_input);
        for (auto ptr : d_batch_pre_activations) {
            gpuCtx_->free_vector(ptr);
        }
        for (auto ptr : d_batch_activations) {
            gpuCtx_->free_vector(ptr);
        }
    }
}

void Network_b::init_batch_buffers(int mini_batch_size) {

    // mini_batch_size (capped at max_batch_size_). This avoids wasting memory for smaller batches.
    if (!batch_buffers_allocated_) {
        if (mini_batch_size > max_batch_size_) {
            throw std::runtime_error("Mini-batch size (" + std::to_string(mini_batch_size) +
                ") exceeds maximum allowed (" + std::to_string(max_batch_size_) + ")");
        }

        allocated_batch_size_ = mini_batch_size;
        d_batch_pre_activations.resize(layers.size());
        d_batch_activations.resize(layers.size());
        d_batch_deltas.resize(layers.size());

        // Allocate main input buffer
        int input_size = sizes[0];
        gpuCtx_->allocate_vector(&d_batch_main_input, input_size * allocated_batch_size_);
        gpuCtx_->set_to_zero(d_batch_main_input, input_size * allocated_batch_size_);

        // Allocate per-layer buffers
        for (size_t i = 0; i < layers.size(); ++i) {
            int layer_size = sizes[i + 1];  // Neurons in this layer
            gpuCtx_->allocate_vector(&d_batch_pre_activations[i], layer_size * allocated_batch_size_);
            gpuCtx_->allocate_vector(&d_batch_activations[i], layer_size * allocated_batch_size_);
            gpuCtx_->allocate_batch_vector(&d_batch_deltas[i], layer_size, allocated_batch_size_);
            gpuCtx_->set_to_zero(d_batch_pre_activations[i], layer_size * allocated_batch_size_);
            gpuCtx_->set_to_zero(d_batch_activations[i], layer_size * allocated_batch_size_);
            gpuCtx_->set_to_zero_batch(d_batch_deltas[i], layer_size, allocated_batch_size_);
        }

        // Allocate targets and temp deriv buffer
        int output_size = sizes.back();
        gpuCtx_->allocate_batch_vector(&d_batch_targets, output_size, allocated_batch_size_);
        gpuCtx_->allocate_batch_vector(&d_temp_deriv_batch, output_size, allocated_batch_size_);
        gpuCtx_->set_to_zero_batch(d_batch_targets, output_size, allocated_batch_size_);
        gpuCtx_->set_to_zero_batch(d_temp_deriv_batch, output_size, allocated_batch_size_);

        batch_buffers_allocated_ = true;  // Mark as allocated
    }
    else {
        throw std::runtime_error(" batch buffers already allocated");
    }
}

/**
 * @brief Trains the network using stochastic gradient descent.
 * Shuffles training data and updates parameters via mini-batches.
 * @param training_data Vector of (input, target) pairs
 * @param epochs Number of training epochs
 * @param mini_batch_size Size of each mini-batch
 * @param eta Learning rate
 * @param test_data Optional test data for evaluation
 * @param verbose If true, display detailed metrics per epoch
 */
void Network_b::SGD(std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>& training_data,
    int epochs, int mini_batch_size, double eta,
    const std::vector<std::pair<Eigen::VectorXd, int>>* test_data,
    bool verbose) {
    size_t n = training_data.size();
    size_t n_test = test_data ? test_data->size() : 0;

    // mini_batch_size (capped at max_batch_size_). This avoids wasting memory for smaller batches.
    if (!batch_buffers_allocated_) {
        init_batch_buffers(mini_batch_size);
    }


    for (int j = 0; j < epochs; ++j) {
        std::shuffle(training_data.begin(), training_data.end(), rng);
        double batch_gradient_norm = 0.0;

        size_t num_batches = (n + mini_batch_size - 1) / mini_batch_size; // Ceiling division

        for (size_t k = 0; k < n; k += mini_batch_size) {
            std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> mini_batch(
                training_data.begin() + k,
                training_data.begin() + std::min(k + mini_batch_size, n));

            batch_gradient_norm += update_mini_batch(mini_batch, eta, n);
        }

        batch_gradient_norm /= num_batches;

        if (verbose && test_data) {
            auto [correct, total_loss] = evaluate_batch(*test_data, n);

            double accuracy = (n_test > 0) ? (correct * 100.0 / n_test) : 0.0;
            double loss = (n_test > 0) ? total_loss / n_test : 0.0;
            std::cout << std::fixed << std::setprecision(4);
            std::cout << "Epoch " << j
                << ": Accuracy = " << accuracy << "%"
                << ", Correct = " << correct << "/" << n_test
                << ", Loss = " << loss
                << ", Gradient Norm = " << batch_gradient_norm;
            if (lambda > 0.0) {
                std::cout << ", Lambda = " << lambda;
            }
            std::cout << std::endl;
        }
        else if (test_data) {
            auto [correct, total_loss] = evaluate_batch(*test_data, n);
            std::cout << "Epoch " << j << ": Correct Predictions = " << correct << "/" << n_test << std::endl;
        }
        else {
            std::cout << "Epoch " << j << " complete" << std::endl;
        }
    }
}

void Network_b::feedforward_batch(const std::vector<Eigen::VectorXd>& batch_inputs, std::vector<Eigen::VectorXd>& batch_outputs) {
    
    if (!batch_buffers_allocated_) {
        init_batch_buffers(batch_inputs.size());
    }

    int batch_size = static_cast<int>(batch_inputs.size());
    if (batch_size == 0) return;
    if (allocated_batch_size_ == 0) {
        allocated_batch_size_ = batch_size;
    }
    if (batch_size > allocated_batch_size_) {
        throw std::runtime_error("Batch size exceeds allocated size; reallocate or use sub-batching");
    }

    // Copy batch inputs to device (column-major: sizes[0] x batch_size)
    gpuCtx_->copy_batch_to_device(d_batch_main_input, batch_inputs, false);  // false: not transposed

    // Chain through layers
    const double* d_current = d_batch_main_input;
    for (size_t i = 0; i < layers.size(); ++i) {
        d_current = layers[i]->forward_gpu_batch(
            d_current,
            d_batch_pre_activations[i],
            d_batch_activations[i],
            batch_size
        );
    }

    // Copy final activations to host
    gpuCtx_->copy_batch_to_host(batch_outputs, d_current, sizes.back(), batch_size);
}


void Network_b::backprop_batch(const std::vector<Eigen::VectorXd>& batch_targets, double& batch_loss) {
   

    int batch_size = static_cast<int>(batch_targets.size());
    if (batch_size > allocated_batch_size_) {
        throw std::runtime_error("Batch size exceeds allocated buffers");
    }

    // Copy targets to device
    gpuCtx_->copy_batch_to_device(d_batch_targets, batch_targets, false);

    // Output layer delta: cost_prime = output - target (for MSE or sigmoid+CE)
    int output_size = sizes.back();
    double* d_output_batch = d_batch_activations.back();

    bool apply_deriv = false;

    if (neuron_type_ == NeuronType::SIGMOID) {
        switch (loss_type_) { // Output layer: compute delta = cost_derivative * (sigmoid' for MSE, 1 for CE)
        case LossType::MSE: {
            apply_deriv = true;
            gpuCtx_->cost_prime_mse_crossent_batched(d_output_batch, d_batch_targets, d_batch_deltas.back(), output_size, batch_size);
            break;
        };
        case LossType::CROSS_ENTROPY: {
            apply_deriv = false;
            gpuCtx_->cost_prime_mse_crossent_batched(d_output_batch, d_batch_targets, d_batch_deltas.back(), output_size, batch_size);
            break;
        };
        default:
            throw std::runtime_error("Unsupported loss type");
            break;
        };
    }
    else {
        throw std::runtime_error("Unsupported neuron type");
    }

    //skip for now
    //{
    //    // Compute loss
    //    if (loss_type_ == LossType::MSE) {
    //        batch_loss = contextGPU_->compute_mse_loss_batch_gpu(d_output_batch, d_batch_targets, output_size, batch_size);
    //    }
    //    else if (loss_type_ == LossType::CROSS_ENTROPY) {
    //        batch_loss = 0.0;  // TODO: Implement batched CE loss
    //        throw std::runtime_error("CE loss not implemented for batch");
    //    }
    //    else {
    //        batch_loss = 0.0;
    //    }
    //}

    // Apply activation derivative for output layer (MSE only; CE skips due to sigmoid cancellation)
    if (apply_deriv) {
        gpuCtx_->computeActivationDerivativeGPU_batch(d_batch_pre_activations.back(), d_temp_deriv_batch, output_size, batch_size, activation_.get());
        gpuCtx_->launch_elementwise_multiply_batch(d_batch_deltas.back(), d_temp_deriv_batch, d_batch_deltas.back(), output_size, batch_size);
    }

    // Compute loss (stub for now; use MSE)
    batch_loss = 0.0;  // TODO: Add batched loss computation (e.g., reuse compute_mse_loss_batch_gpu)

    // Propagate deltas backward (from second-last to first)
    for (int l = static_cast<int>(layers.size()) - 2; l >= 0; --l) {
        Layer_b* next_layer = layers[l + 1].get();

        // delta_l = W_{l+1}^T * delta_{l+1}
        int m_next = next_layer->get_num_neurons();
        int n_next = next_layer->get_num_inputs();

        // Pass (m, n) in correct order
        gpuCtx_->compute_delta_back_batch(next_layer->get_d_weights(), d_batch_deltas[l + 1],
            d_batch_deltas[l], m_next, n_next, batch_size);

        // Apply activation derivative (always for hidden layers)
        gpuCtx_->computeActivationDerivativeGPU_batch(d_batch_pre_activations[l], d_temp_deriv_batch, sizes[l + 1], batch_size, activation_.get());
        gpuCtx_->launch_elementwise_multiply_batch(d_batch_deltas[l], d_temp_deriv_batch, d_batch_deltas[l], sizes[l + 1], batch_size);
    }

    // Compute gradients (forward order)
    const double* d_prev_a = d_batch_main_input;
    for (size_t l = 0; l < layers.size(); ++l) {
        gpuCtx_->computeGradientsGPU_batch(d_batch_deltas[l], d_prev_a, accumulate_weight_grads[l], accumulate_bias_grads[l], sizes[l + 1], sizes[l], batch_size);
        d_prev_a = d_batch_activations[l];
    }
}

double Network_b::update_mini_batch(
    const std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>& mini_batch,
    double eta, size_t n) {
  
    if (!batch_buffers_allocated_) {
        init_batch_buffers(mini_batch.size());
    }

    int batch_size = static_cast<int>(mini_batch.size());
    if (batch_size == 0) return 0.0;

    // Zero accumulators for next batch
    for (size_t i = 0; i < layers.size(); ++i) {
        // Zero accumulators for next batch
        gpuCtx_->set_to_zero(accumulate_weight_grads[i], weight_rows[i] * weight_cols[i]);
        gpuCtx_->set_to_zero(accumulate_bias_grads[i], bias_sizes[i]);
    }

    // Prepare inputs and targets
    std::vector<Eigen::VectorXd> batch_inputs(batch_size);
    std::vector<Eigen::VectorXd> batch_targets(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        batch_inputs[i] = mini_batch[i].first;
        batch_targets[i] = mini_batch[i].second;
    }

    // Forward pass
    std::vector<Eigen::VectorXd> batch_outputs;
    feedforward_batch(batch_inputs, batch_outputs);

    // Backprop: Accumulates grads and computes loss
    double batch_loss;
    backprop_batch(batch_targets, batch_loss);

    // Scale gradients and apply L2 regularization
    double scale = eta / batch_size;
    double reg_scale = lambda * mini_batch.size() / n;
    //double reg_scale = lambda / n;

    for (size_t i = 0; i < layers.size(); ++i) {
        // Add L2 regularization: weight_grad += lambda * weights
        if (lambda > 0.0) {
            gpuCtx_->add_regularization(accumulate_weight_grads[i], layers[i]->get_d_weights(), reg_scale, weight_rows[i], weight_cols[i]);
        }

        // Update parameters
        layers[i]->update_parameters(accumulate_weight_grads[i], accumulate_bias_grads[i],
            temp_weight_grads[i], temp_bias_grads[i], scale);
    }

    // Compute gradient norm (already accumulated)
    return gpuCtx_->compute_gradient_norm_gpu(accumulate_weight_grads, accumulate_bias_grads,
        weight_rows, weight_cols, bias_sizes, batch_size);
}

// Updated evaluate_batch overload for VectorXd targets in Network.cpp
// Changes: Added sub-batching to handle large test_data > allocated_batch_size_
// Computed weight_norm once outside loops
// No other errors found; loss computation uses host-side loops (inefficient but correct)
// TODO: Optimize with batched GPU loss computation in future phases
std::pair<int, double> Network_b::evaluate_batch(const std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>& test_data, size_t n) {
    int correct = 0;
    double total_loss = 0.0;
    double weight_norm = 0.0;

    size_t test_size = test_data.size();
    if (test_size == 0) {
        throw std::runtime_error("test data size zero");
    }

    if (!batch_buffers_allocated_) {
        init_batch_buffers(max_batch_size_);
    }

    // Compute weight_norm once
    for (const auto& layer : layers) {
        weight_norm += gpuCtx_->compute_squared_normGPU(layer->get_weights());
    }

    int max_sub_batch = allocated_batch_size_;

    for (size_t start = 0; start < test_size; start += max_sub_batch) {
        size_t sub_size = std::min(static_cast<size_t>(max_sub_batch), test_size - start);

        // Extract sub-batch data
        std::vector<Eigen::VectorXd> sub_inputs(sub_size);
        std::vector<Eigen::VectorXd> sub_targets(sub_size);
        for (size_t i = 0; i < sub_size; ++i) {
            sub_inputs[i] = test_data[start + i].first;
            sub_targets[i] = test_data[start + i].second;
        }

        // Forward pass on sub-batch
        std::vector<Eigen::VectorXd> sub_outputs;
        feedforward_batch(sub_inputs, sub_outputs);

        // Copy targets to device for batched loss
        gpuCtx_->copy_batch_to_device(d_batch_targets, sub_targets, false);

        // Compute batched loss on GPU
        double sub_loss = 0.0;
        if (neuron_type_ == NeuronType::SIGMOID && loss_type_ == LossType::MSE) {
            sub_loss = gpuCtx_->compute_mse_loss_batchGPU(d_batch_activations.back(), d_batch_targets, sizes.back(), sub_size);
        }
        else if (neuron_type_ == NeuronType::SIGMOID && loss_type_ == LossType::CROSS_ENTROPY) {
            sub_loss = gpuCtx_->compute_cross_entropy_loss_batchGPU(d_batch_activations.back(), d_batch_targets, sizes.back(), sub_size);
        }
        else {
            throw std::runtime_error("Unsupported loss type");
        }
        total_loss += sub_loss;

        // Accumulate correct (keep host-side for argmax; small batch sizes make this efficient)
        for (size_t i = 0; i < sub_size; ++i) {
            if (is_correct_prediction(sub_outputs[i], sub_targets[i])) {
                ++correct;
            }
        }

        {
            //old code
            // Accumulate correct and loss
            /*for (size_t i = 0; i < sub_size; ++i) {
                if (is_correct_prediction(sub_outputs[i], sub_targets[i])) {
                    ++correct;
                }

                if (neuron_type_ == NeuronType::SIGMOID && loss_type_ == LossType::MSE) {
                    total_loss += gpuCtx_->compute_mse_lossGPU(sub_outputs[i], sub_targets[i]);
                }
                else if (neuron_type_ == NeuronType::SIGMOID && loss_type_ == LossType::CROSS_ENTROPY) {
                    total_loss += gpuCtx_->compute_cross_entropy_lossGPU(sub_outputs[i], sub_targets[i]);
                }
                else {
                    throw std::runtime_error("Unsupported loss type");
                }
            }*/
        }
    }

    if (lambda > 0.0 && n > 0) {
        total_loss += 0.5 * lambda * weight_norm / n;
    }

    last_test_loss = total_loss;
    return { correct, total_loss };
}

// Updated evaluate_batch overload for int labels in Network.cpp
// Changes: Added sub-batching similar to above
// Used label_to_one_hot helper for targets
// Used int version of is_correct_prediction for efficiency (avoids unnecessary argmax on one-hot)
// No other errors found; consistent with the other overload
std::pair<int, double> Network_b::evaluate_batch(const std::vector<std::pair<Eigen::VectorXd, int>>& test_data, size_t n) {
    int correct = 0;
    double total_loss = 0.0;
    double weight_norm = 0.0;

    size_t test_size = test_data.size();
    if (test_size == 0) {
        throw std::runtime_error("test data size zero");
    }

    if (!batch_buffers_allocated_) {
        init_batch_buffers(max_batch_size_);
    }

    // Compute weight_norm once
    for (const auto& layer : layers) {
        weight_norm += gpuCtx_->compute_squared_normGPU(layer->get_weights());
    }

    int max_sub_batch = allocated_batch_size_;
    for (size_t start = 0; start < test_size; start += max_sub_batch) {
        size_t sub_size = std::min(static_cast<size_t>(max_sub_batch), test_size - start);

        // Extract sub-batch data
        std::vector<Eigen::VectorXd> sub_inputs(sub_size);
        std::vector<Eigen::VectorXd> sub_targets(sub_size);
        std::vector<int> sub_labels(sub_size);
        for (size_t i = 0; i < sub_size; ++i) {
            sub_inputs[i] = test_data[start + i].first;
            sub_labels[i] = test_data[start + i].second;
            sub_targets[i] = label_to_one_hot(sub_labels[i]);
        }

        // Forward pass on sub-batch
        std::vector<Eigen::VectorXd> sub_outputs;
        feedforward_batch(sub_inputs, sub_outputs);

        // Copy targets to device for batched loss (sub_targets are one-hot)
        gpuCtx_->copy_batch_to_device(d_batch_targets, sub_targets, false);

        // Compute batched loss on GPU
        double sub_loss = 0.0;
        if (neuron_type_ == NeuronType::SIGMOID && loss_type_ == LossType::MSE) {
            sub_loss = gpuCtx_->compute_mse_loss_batchGPU(d_batch_activations.back(), d_batch_targets, sizes.back(), sub_size);
        }
        else if (neuron_type_ == NeuronType::SIGMOID && loss_type_ == LossType::CROSS_ENTROPY) {
            sub_loss = gpuCtx_->compute_cross_entropy_loss_batchGPU(d_batch_activations.back(), d_batch_targets, sizes.back(), sub_size);
        }
        else {
            throw std::runtime_error("Unsupported loss type");
        }
        total_loss += sub_loss;

        // Accumulate correct (keep host-side for argmax; small batch sizes make this efficient)
        for (size_t i = 0; i < sub_size; ++i) {
            if (is_correct_prediction(sub_outputs[i], sub_labels[i])) {
                ++correct;
            }
        }

        {
            // Accumulate correct and loss
            /*for (size_t i = 0; i < sub_size; ++i) {
                if (is_correct_prediction(sub_outputs[i], sub_labels[i])) {
                    ++correct;
                }

                if (neuron_type_ == NeuronType::SIGMOID && loss_type_ == LossType::MSE) {
                    total_loss += gpuCtx_->compute_mse_lossGPU(sub_outputs[i], sub_targets[i]);
                }
                else if (neuron_type_ == NeuronType::SIGMOID && loss_type_ == LossType::CROSS_ENTROPY) {
                    total_loss += gpuCtx_->compute_cross_entropy_lossGPU(sub_outputs[i], sub_targets[i]);
                }
                else {
                    throw std::runtime_error("Unsupported loss type");
                }
            }*/
        }
    }

    if (lambda > 0.0 && n > 0) {
        total_loss += 0.5 * lambda * weight_norm / n;
    }

    last_test_loss = total_loss;
    return { correct, total_loss };
}

/**
 * @brief Sets the weights of a specific layer.
 * @param layer_idx Index of the layer
 * @param weights New weight matrix
 */
void Network_b::set_layer_weights(size_t layer_idx, const Eigen::MatrixXd& weights) {
    if (layer_idx >= layers.size()) {
        throw std::out_of_range("Layer index out of bounds");
    }
    if (weights.rows() != layers[layer_idx]->get_num_neurons() || weights.cols() != layers[layer_idx]->get_num_inputs()) {
        throw std::invalid_argument("Weight matrix dimensions mismatch");
    }
    layers[layer_idx]->set_weights(weights);
}

/**
 * @brief Sets the biases of a specific layer.
 * @param layer_idx Index of the layer
 * @param biases New bias vector
 */
void Network_b::set_layer_biases(size_t layer_idx, const Eigen::VectorXd& biases) {
    if (layer_idx >= layers.size()) {
        throw std::out_of_range("Layer index out of bounds");
    }
    if (biases.size() != layers[layer_idx]->get_num_neurons()) {
        throw std::invalid_argument("Bias vector dimension mismatch");
    }
    layers[layer_idx]->set_biases(biases);
}

bool Network_b::is_correct_prediction(const Eigen::VectorXd& output, int label) {
    Eigen::Index predicted;
    output.maxCoeff(&predicted);
    return predicted == static_cast<Eigen::Index>(label);
}

bool Network_b::is_correct_prediction(const Eigen::VectorXd& output, const Eigen::VectorXd& target) {
    Eigen::Index predicted, actual;
    output.maxCoeff(&predicted);
    target.maxCoeff(&actual);
    return predicted == actual;
}

Eigen::VectorXd Network_b::label_to_one_hot(int label) const {
    Eigen::VectorXd vec = Eigen::VectorXd::Zero(sizes.back());
    vec(label) = 1.0;
    return vec;
}