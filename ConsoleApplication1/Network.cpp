#include "Network.hpp"
#include "CPUComputationContext.hpp"
#include "GPUComputationContext.hpp"
#include"utils.h"
#include <iomanip>


/**
 * @brief Constructs a network with specified layer sizes.
 * Initializes layers with Xavier-initialized weights and biases.
 * @param sizes Vector of layer sizes (e.g., {784, 30, 10} for MNIST)
 */
Network::Network(const std::vector<int>& sizes, double lambda, LossType loss_type, NeuronType neuron_type, ComputationContext* context, unsigned int seed)
    : 
    sizes(sizes), num_layers(sizes.size()), 
    rng(seed), last_test_loss(0.0), lambda(lambda), 
    loss_type_(loss_type), neuron_type_(neuron_type), 
    context_(context), owns_context_(context == nullptr), 
    is_gpu_context_(dynamic_cast<GPUComputationContext*>(context_) != nullptr) {

    if (!context_) {
        context_ = new CPUComputationContext();
        is_gpu_context_ = false;
    }

    if (is_gpu_context_) {
        contextGPU_ = dynamic_cast<GPUComputationContext*>(context_);
    }
    else {
        contextCPU_ = dynamic_cast<CPUComputationContext*>(context_);
    }

    // Dynamically create activation based on neuron_type
    switch (neuron_type_) {
        case NeuronType::SIGMOID:
            activation_ = std::make_unique<SigmoidActivation>();
            break;
        default:
            throw std::runtime_error("Unsupported neuron type");
    }

    for (size_t i = 1; i < sizes.size(); ++i) {
        layers.emplace_back(std::make_unique<Layer>(
            sizes[i - 1], sizes[i], activation_.get(), context_, static_cast<unsigned int>(rng())
        ));
    }

    //Initialize GPU storage pointers if gpu context
    if (is_gpu_context_) {
      
        //Initialize device memory pointers
        for (size_t i = 0; i < sizes.size() - 1; ++i) {
            double* weightGrad_currentLayer = nullptr;
            double* biasGrad_currentLayer = nullptr;
            double* temp_weightGrad = nullptr;
            double* temp_biasGrad = nullptr;

            contextGPU_->allocate_weights(&weightGrad_currentLayer, layers[i]->get_num_neurons(), layers[i]->get_num_inputs());
            contextGPU_->allocate_biases(&biasGrad_currentLayer, layers[i]->get_num_neurons());
            contextGPU_->allocate_weights(&temp_weightGrad, layers[i]->get_num_neurons(), layers[i]->get_num_inputs());
            contextGPU_->allocate_biases(&temp_biasGrad, layers[i]->get_num_neurons());

            Eigen::MatrixXd zeroMatrix = Eigen::MatrixXd::Zero(layers[i]->get_num_neurons(), layers[i]->get_num_inputs());
            Eigen::VectorXd zeroVec = Eigen::VectorXd::Zero(layers[i]->get_num_neurons());

            contextGPU_->copy_to_device(weightGrad_currentLayer, zeroMatrix);
            contextGPU_->copy_to_device(biasGrad_currentLayer, zeroVec);
            contextGPU_->copy_to_device(temp_weightGrad, zeroMatrix);
            contextGPU_->copy_to_device(temp_biasGrad, zeroVec);

            accumulate_weight_grads.push_back(weightGrad_currentLayer);
            accumulate_bias_grads.push_back(biasGrad_currentLayer);
            temp_weight_grads.push_back(temp_weightGrad);
            temp_bias_grads.push_back(temp_biasGrad);

            weight_rows.push_back(layers[i]->get_num_neurons());
            weight_cols.push_back(layers[i]->get_num_inputs());
            bias_sizes.push_back(layers[i]->get_num_neurons());
        }
    }

}

Network::~Network() {
    if (is_gpu_context_) {
        for (auto ptr : accumulate_weight_grads) {
            contextGPU_->free_weights(ptr);
        }
        for (auto ptr : accumulate_bias_grads) {
            contextGPU_->free_biases(ptr);
        }
        for (auto ptr : temp_weight_grads) {
            contextGPU_->free_weights(ptr);
        }
        for (auto ptr : temp_bias_grads) {
            contextGPU_->free_biases(ptr);
        }
    }

    if (owns_context_ && context_) {
        delete context_;
    }
}

/**
 * @brief Computes the network output for a given input.
 * Passes the input through each layer's forward pass.
 * @param a Input vector
 * @return Output activations of the final layer
 */
Eigen::VectorXd Network::feedforward(const Eigen::VectorXd& a) {
    Eigen::VectorXd activation = a;
    for (auto& layer : layers) {
        activation = layer->forward(activation);
    }
    return activation;
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
void Network::SGD(std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>& training_data,
    int epochs, int mini_batch_size, double eta,
    const std::vector<std::pair<Eigen::VectorXd, int>>* test_data, 
    bool verbose) {
    size_t n = training_data.size();
    size_t n_test = test_data ? test_data->size() : 0;
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
            auto [correct, total_loss] = evaluate(*test_data, n);
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
            auto [correct, total_loss] = evaluate(*test_data, n);
            std::cout << "Epoch " << j << ": Correct Predictions = " << correct << "/" << n_test << std::endl;
        }
        else {
            std::cout << "Epoch " << j << " complete" << std::endl;
        }
    }
}


double Network::update_mini_batch(const std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>& mini_batch, double eta, size_t n) {
    if (mini_batch.empty()) return 0.0;
    double norm = 0.0;

    if (is_gpu_context_) {
        // GPU path: Accumulate gradients
        std::vector<double*> weight_grads_acc = accumulate_weight_grads;
        std::vector<double*> bias_grads_acc = accumulate_bias_grads;

        // Zero out accumulators
        for (size_t i = 0; i < layers.size(); ++i) {
            contextGPU_->set_to_zero(weight_grads_acc[i], weight_rows[i] * weight_cols[i]);
            contextGPU_->set_to_zero(bias_grads_acc[i], bias_sizes[i]);
        }

        // Accumulate gradients over mini-batch
        for (const auto& [x, y] : mini_batch) {
            auto [nabla_b, nabla_w] = backprop_gpu(x, y, n);
            contextGPU_->accumulateGradientsGPU(nabla_w, nabla_b, weight_grads_acc, bias_grads_acc, weight_rows, weight_cols, bias_sizes, 1.0);

        }

        // Add L2 regularization
        if (lambda > 0.0) {
            double reg_scale = lambda * mini_batch.size() / n;
            for (size_t i = 0; i < layers.size(); ++i) {
                contextGPU_->add_regularization(accumulate_weight_grads[i], layers[i]->get_d_weights(), reg_scale, weight_rows[i], weight_cols[i]);
            }
        }

        // Update parameters
        double update_scale = eta / mini_batch.size();
        for (size_t i = 0; i < layers.size(); ++i) {
            layers[i]->update_parameters_gpu(accumulate_weight_grads[i], accumulate_bias_grads[i], temp_weight_grads[i], temp_bias_grads[i], update_scale);
        }

        // Compute gradient norm
        norm = contextGPU_->compute_gradient_norm_gpu(accumulate_weight_grads, accumulate_bias_grads, weight_rows, weight_cols, bias_sizes, mini_batch.size());
    }
    else {
        // CPU path 
        std::vector<Eigen::MatrixXd> weight_grads(layers.size());
        std::vector<Eigen::VectorXd> bias_grads(layers.size());

        for (size_t i = 0; i < layers.size(); ++i) {
            weight_grads[i] = Eigen::MatrixXd::Zero(layers[i]->get_num_neurons(), layers[i]->get_num_inputs());
            bias_grads[i] = Eigen::VectorXd::Zero(layers[i]->get_num_neurons());
        }

        for (const auto& [x, y] : mini_batch) {
            auto [nabla_b, nabla_w] = backprop_cpu(x, y, n);
            contextCPU_->accumulateGradientsCPU(nabla_w, nabla_b, weight_grads, bias_grads, 1.0);
        }

        // Add L2 regularization
        if (lambda > 0.0) {
            double reg_scale = lambda * mini_batch.size() / n;
            for (size_t i = 0; i < layers.size(); ++i) {
                weight_grads[i] += reg_scale * layers[i]->get_weights();
            }
        }

        double update_scale = eta / mini_batch.size();
        for (size_t i = 0; i < layers.size(); ++i) {
            layers[i]->update_parameters_cpu(weight_grads[i], bias_grads[i], update_scale);
        }

        double total_sq_norm = 0.0;
        for (size_t i = 0; i < layers.size(); ++i) {
            total_sq_norm += contextCPU_->compute_squared_normCPU(weight_grads[i]);
            total_sq_norm += contextCPU_->compute_squared_normCPU(bias_grads[i]);
        }
        norm = std::sqrt(total_sq_norm) / mini_batch.size();
    }

    return norm;
}


std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::MatrixXd>> Network::backprop_cpu(
    const Eigen::VectorXd& x, const Eigen::VectorXd& y, size_t n) {

    feedforward(x);  // Compute and cache activations for all layers

    std::vector<Eigen::VectorXd> nabla_b(layers.size());
    std::vector<Eigen::MatrixXd> nabla_w(layers.size());

    // Output layer: compute delta = cost_derivative * (sigmoid' for MSE, 1 for CE)
    Eigen::VectorXd activation = layers.back()->get_activations();
    Eigen::VectorXd cost_prime = cost_derivative(activation, y);  // output - y

    bool apply_deriv;
    switch (loss_type_) {
    case LossType::MSE: {
        apply_deriv = true;
        break;
    };
    case LossType::CROSS_ENTROPY: {
        apply_deriv = false;
        break;
    };
    default:
        throw std::runtime_error("Unsupported loss type");
        break;
    };

    //Output layer gradients
    layers.back()->compute_gradients_cpu(cost_prime, nabla_w.back(), nabla_b.back(), apply_deriv);

    // Hidden layers: backpropagate deltas
    Eigen::VectorXd delta = nabla_b.back();
    for (int l = layers.size() - 2; l >= 0; --l) {
        auto& layer = layers[l];
        auto& next_layer = layers[l + 1];
        Eigen::VectorXd incoming_delta = next_layer->get_weights().transpose() * delta;
        layer->compute_gradients_cpu(incoming_delta, nabla_w[l], nabla_b[l], true);  // Always apply derivative for hidden layers
        delta = nabla_b[l];
    }

    return { nabla_b, nabla_w };
}

std::pair<std::vector<double*>, std::vector<double*>> Network::backprop_gpu(
    const Eigen::VectorXd& x, const Eigen::VectorXd& y, size_t n) {

    feedforward(x);  // Sets device and host activations

    std::vector<double*> nabla_b(layers.size());
    std::vector<double*> nabla_w(layers.size());

    // Allocate device memory for y
    double* d_y = nullptr;
    contextGPU_->allocate_vector(&d_y, sizes.back());
    contextGPU_->copy_to_device(d_y, y);

    // Reset per-example gradients and deltas for all layers
    for (auto& layer : layers) {
        contextGPU_->set_to_zero(layer->get_d_weight_grads_(), layer->get_num_neurons() * layer->get_num_inputs());
        contextGPU_->set_to_zero(layer->get_d_delta_(), layer->get_num_neurons());
    }

    //TODO: move Output layer: delta calc in a switch based on cost function(loss_type) type.
    // Output layer: delta = output - y (for both MSE and CE)
    auto output_layer = layers.back().get();
    double* d_cost_prime = nullptr;
    contextGPU_->allocate_vector(&d_cost_prime, output_layer->get_num_neurons());
    contextGPU_->launch_elementwise_subtract(output_layer->get_d_activations_(), d_y, d_cost_prime, output_layer->get_num_neurons());

    bool apply_deriv;
    switch (loss_type_) {
    case LossType::MSE: {
        apply_deriv = true;
        break;
    };
    case LossType::CROSS_ENTROPY: {
        apply_deriv = false;
        break;
    };
    default:
        throw std::runtime_error("Unsupported loss type");
        break;
    };

    // Compute gradients for output layer
    output_layer->compute_gradients_gpu(d_cost_prime, apply_deriv);
    nabla_w.back() = output_layer->get_d_weight_grads_();
    nabla_b.back() = output_layer->get_d_delta_();

    // Free temporary device memory
    contextGPU_->free_vector(d_cost_prime);
    contextGPU_->free_vector(d_y);

    // Hidden layers: backpropagate deltas
    for (int l = static_cast<int>(layers.size()) - 2; l >= 0; --l) {
        auto layer = layers[l].get();
        auto next_layer = layers[l + 1].get();
        double* d_incoming_delta = nullptr;
        contextGPU_->allocate_vector(&d_incoming_delta, layer->get_num_neurons());

        //Compute delta back (delta = W^T * delta_next)
        contextGPU_->compute_delta_back(next_layer->get_d_weights(), next_layer->get_d_delta_(), d_incoming_delta,
            next_layer->get_num_neurons(), next_layer->get_num_inputs());

        layer->compute_gradients_gpu(d_incoming_delta, true);  // Always apply derivative for hidden layers
        nabla_w[l] = layer->get_d_weight_grads_();
        nabla_b[l] = layer->get_d_delta_();
        contextGPU_->free_vector(d_incoming_delta);
    }

    return { nabla_b, nabla_w };  // Return device pointers
}


// Dispatcher
std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::MatrixXd>> Network::backprop(
    const Eigen::VectorXd& x, const Eigen::VectorXd& y, size_t n) {
    if (is_gpu_context_) {
        auto [nabla_b, nabla_w] = backprop_gpu(x, y, n);
        // Convert device pointers to host for compatibility
        std::vector<Eigen::VectorXd> host_nabla_b;
        std::vector<Eigen::MatrixXd> host_nabla_w;
        for (size_t i = 0; i < layers.size(); ++i) {
            Eigen::VectorXd b = Eigen::VectorXd::Zero(layers[i]->get_num_neurons());
            Eigen::MatrixXd w = Eigen::MatrixXd::Zero(layers[i]->get_num_neurons(), layers[i]->get_num_inputs());
            contextGPU_->copy_to_host(b, nabla_b[i], layers[i]->get_num_neurons());
            contextGPU_->copy_to_host(w, nabla_w[i], layers[i]->get_num_neurons(), layers[i]->get_num_inputs());
            host_nabla_b.push_back(b);
            host_nabla_w.push_back(w);
        }
        freeDevicePointers(nabla_b);
        freeDevicePointers(nabla_w);
        return { host_nabla_b, host_nabla_w };
    }
    else {
        return backprop_cpu(x, y, n);
    }
}

/**
 * @brief Evaluates the network on test data and computes loss.
 * @param test_data Vector of (input, label) pairs
 * @param n Number of training examples for L2 regularization scaling
 * @return Pair of (correct predictions, total MSE loss including regularization)
 */
std::pair<int, double> Network::evaluate(const std::vector<std::pair<Eigen::VectorXd, int>>& test_data, size_t n) {
    int correct = 0;
    double total_loss = 0.0;
    double weight_norm = 0.0;

    if (is_gpu_context_) {
        
        for (const auto& layer : layers) {
            weight_norm += contextGPU_->compute_squared_normGPU(layer->get_weights());
        }

        for (const auto& [x, y] : test_data) {
            Eigen::VectorXd output = feedforward(x);

            if (is_correct_prediction(output, y))
                ++correct;

            Eigen::VectorXd target = Eigen::VectorXd::Zero(output.size());
            target(y) = 1.0; // One-hot encoding for target label

            //TODO: put a switch here
            if (neuron_type_ == NeuronType::SIGMOID && loss_type_ == LossType::MSE) {
                total_loss += contextGPU_->compute_mse_lossGPU(output, target);
            }
            else if (neuron_type_ == NeuronType::SIGMOID && loss_type_ == LossType::CROSS_ENTROPY) {
                total_loss += contextGPU_->compute_cross_entropy_lossGPU(output, target);
            }
            else {
                throw std::runtime_error("Unsupported loss type");
            }
        }
        if (lambda > 0.0 && n > 0) {
            total_loss += 0.5 * lambda * weight_norm / n; // Scaled L2 regularization
        }

        last_test_loss = total_loss; // Cache total loss
        return { correct, total_loss };
    }
    else {
        
        for (const auto& layer : layers) {
            weight_norm += contextCPU_->compute_squared_normCPU(layer->get_weights());
        }

        for (const auto& [x, y] : test_data) {
            Eigen::VectorXd output = feedforward(x);

            if (is_correct_prediction(output, y))
                ++correct;

            Eigen::VectorXd target = Eigen::VectorXd::Zero(output.size());
            target(y) = 1.0; // One-hot encoding for target label

            //TODO: put a switch here
            if (neuron_type_ == NeuronType::SIGMOID && loss_type_ == LossType::MSE) {
                total_loss += contextCPU_->compute_mse_lossCPU(output, target);
            }
            else if (neuron_type_ == NeuronType::SIGMOID && loss_type_ == LossType::CROSS_ENTROPY) {
                total_loss += contextCPU_->compute_cross_entropy_lossCPU(output, target);
            }
            else {
                throw std::runtime_error("Unsupported loss type");
            }
        }
        if (lambda > 0.0 && n > 0) {
            total_loss += 0.5 * lambda * weight_norm / n; // Scaled L2 regularization
        }

        last_test_loss = total_loss; // Cache total loss
        return { correct, total_loss };
    }

    
}


std::pair<int, double> Network::evaluate(const std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>& test_data, size_t n) {
    int correct = 0;
    double total_loss = 0.0;
    double weight_norm = 0.0;

    if (is_gpu_context_) {
        for (const auto& layer : layers) {
            weight_norm += contextGPU_->compute_squared_normGPU(layer->get_weights());
        }

        for (const auto& [x, y] : test_data) {
            Eigen::VectorXd output = feedforward(x);

            if (is_correct_prediction(output, y))
                ++correct;

            //TODO: add a switch here
            if (neuron_type_ == NeuronType::SIGMOID && loss_type_ == LossType::MSE) {
                total_loss += contextGPU_->compute_mse_lossGPU(output, y);
            }
            else if (neuron_type_ == NeuronType::SIGMOID && loss_type_ == LossType::CROSS_ENTROPY) {
                total_loss += contextGPU_->compute_cross_entropy_lossGPU(output, y);
            }
            else {
                throw std::runtime_error("Unsupported loss type");
            }
        }

        if (lambda > 0.0 && n > 0) {
            total_loss += 0.5 * lambda * weight_norm / n;
        }

        last_test_loss = total_loss;
        return { correct, total_loss };
    }
    else {
        for (const auto& layer : layers) {
            weight_norm += contextCPU_->compute_squared_normCPU(layer->get_weights());
        }

        for (const auto& [x, y] : test_data) {
            Eigen::VectorXd output = feedforward(x);

            if (is_correct_prediction(output, y))
                ++correct;

            //TODO: add a switch here
            if (neuron_type_ == NeuronType::SIGMOID && loss_type_ == LossType::MSE) {
                total_loss += contextCPU_->compute_mse_lossCPU(output, y);
            }
            else if (neuron_type_ == NeuronType::SIGMOID && loss_type_ == LossType::CROSS_ENTROPY) {
                total_loss += contextCPU_->compute_cross_entropy_lossCPU(output, y);
            }
            else {
                throw std::runtime_error("Unsupported loss type");
            }
        }

        if (lambda > 0.0 && n > 0) {
            total_loss += 0.5 * lambda * weight_norm / n;
        }

        last_test_loss = total_loss;
        return { correct, total_loss };
    }
    
}


/**
 * @brief Computes the derivative of the cost function w.r.t. output activations.
 * @param output_activations Output activations of the final layer
 * @param y Target vector
 * @return Cost derivative
 */
Eigen::VectorXd Network::cost_derivative(const Eigen::VectorXd& output_activations, const Eigen::VectorXd& y) const {
    if (neuron_type_ == NeuronType::SIGMOID && (loss_type_ == LossType::MSE || loss_type_ == LossType::CROSS_ENTROPY)) {
        return output_activations - y;  // Common derivative for sigmoid with MSE or CE
    }
    throw std::runtime_error("Unsupported neuron type or loss function combination");
}

/**
 * @brief Computes the mean squared error loss over test data.
 * @param test_data Vector of (input, label) pairs
 * @return Average MSE loss
 */
//double Network::compute_test_loss(const std::vector<std::pair<Eigen::VectorXd, int>>& test_data)
//{
//    double total_loss = 0.0;
//    for (const auto& [x, y] : test_data) {
//        Eigen::VectorXd output = feedforward(x);
//        Eigen::VectorXd target = Eigen::VectorXd::Zero(output.size());
//        target(y) = 1.0; // One-hot encoding for target label
//        Eigen::VectorXd diff = output - target;
//        total_loss += diff.squaredNorm();
//    }
//    return total_loss / test_data.size();
//
//}


/**
 * @brief Computes the L2 norm of gradients for a mini-batch.
 * @param mini_batch Vector of (input, target) pairs
 * @param n Number of training examples for L2 regularization scaling
 * @return L2 norm of gradients
 */
//double Network::compute_gradient_norm(const std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>& mini_batch, size_t n)
//{
//    std::vector<Eigen::MatrixXd> weight_grads;
//    std::vector<Eigen::VectorXd> bias_grads;
//    for (size_t i = 0; i < layers.size(); ++i) {
//        weight_grads.emplace_back(Eigen::MatrixXd::Zero(layers[i]->get_num_neurons(), layers[i]->get_num_inputs()));
//        bias_grads.emplace_back(Eigen::VectorXd::Zero(layers[i]->get_num_neurons()));
//    }
//
//    for (const auto& [x, y] : mini_batch) {
//        auto [delta_nabla_b, delta_nabla_w] = backprop(x, y, n);
//        for (size_t i = 0; i < layers.size(); ++i) {
//            bias_grads[i] += delta_nabla_b[i];
//            weight_grads[i] += delta_nabla_w[i];
//        }
//    }
//
//    double norm = 0.0;
//    for (size_t i = 0; i < layers.size(); ++i) {
//        norm += bias_grads[i].squaredNorm();
//        norm += weight_grads[i].squaredNorm();
//    }
//    return std::sqrt(norm / mini_batch.size());
//}

/**
 * @brief Displays biases for all layers.
 */
void Network::display_biases() const {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== Biases ===" << std::endl;
    for (size_t i = 0; i < layers.size(); ++i) {
        std::string layer_name = (i == layers.size() - 1) ? "Output Layer" : "Hidden Layer " + std::to_string(i + 1);
        std::cout << layer_name << ":\n" << layers[i]->print_parameters(false);
    }
}

/**
 * @brief Displays weights for all layers.
 */
void Network::display_weights() const {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== Weights ===" << std::endl;
    for (size_t i = 0; i < layers.size(); ++i) {
        std::string from_layer = (i == 0) ? "Input Layer" : "Hidden Layer " + std::to_string(i);
        std::string to_layer = (i == layers.size() - 1) ? "Output Layer" : "Hidden Layer " + std::to_string(i + 1);
        std::cout << "From " << from_layer << " to " << to_layer << ":\n" << layers[i]->print_parameters(false);
    }
}

/**
 * @brief Displays biases layer-wise with truncation.
 * @param max_elements Maximum elements to display per layer
 */
void Network::display_layer_biases(int max_elements) const {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== Layer Biases ===" << std::endl;
    for (size_t i = 0; i < layers.size(); ++i) {
        std::string layer_name = (i == layers.size() - 1) ? "Output Layer" : "Hidden Layer " + std::to_string(i + 1);
        std::cout << layer_name << " (" << layers[i]->get_num_neurons() << " biases):" << std::endl;
        const Eigen::VectorXd& biases = layers[i]->get_biases();
        int display_count = std::min(static_cast<int>(biases.size()), max_elements);
        std::cout << "  [";
        for (int j = 0; j < display_count; ++j) {
            std::cout << biases(j);
            if (j < display_count - 1) std::cout << ", ";
        }
        if (display_count < biases.size()) std::cout << ", ...";
        std::cout << "]" << std::endl;
        if (display_count < biases.size()) {
            std::cout << "  (Truncated, total " << biases.size() << " biases)" << std::endl;
        }
        std::cout << std::endl;
    }
}

/**
 * @brief Displays weights layer-wise with truncation.
 * @param max_elements Maximum elements to display per layer
 */
void Network::display_layer_weights(int max_elements) const {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== Layer Weights ===" << std::endl;
    for (size_t i = 0; i < layers.size(); ++i) {
        std::string from_layer = (i == 0) ? "Input Layer" : "Hidden Layer " + std::to_string(i);
        std::string to_layer = (i == layers.size() - 1) ? "Output Layer" : "Hidden Layer " + std::to_string(i + 1);
        std::cout << "From " << from_layer << " to " << to_layer
            << " (" << layers[i]->get_num_neurons() << "x" << layers[i]->get_num_inputs() << " matrix):" << std::endl;
        const Eigen::MatrixXd& weights = layers[i]->get_weights();
        int max_rows = std::min(static_cast<int>(weights.rows()), max_elements);
        int max_cols = std::min(static_cast<int>(weights.cols()), max_elements);
        for (int r = 0; r < max_rows; ++r) {
            std::cout << "  [";
            for (int c = 0; c < max_cols; ++c) {
                std::cout << weights(r, c);
                if (c < max_cols - 1) std::cout << ", ";
            }
            if (max_cols < weights.cols()) std::cout << ", ...";
            std::cout << "]" << std::endl;
        }
        if (max_rows < weights.rows() || max_cols < weights.cols()) {
            std::cout << "  (Truncated, full size: " << weights.rows() << "x" << weights.cols() << ")" << std::endl;
        }
        std::cout << std::endl;
    }
}

/**
 * @brief Displays gradients computed by backpropagation for a single example.
 * @param x Input vector
 * @param y Target vector
 * @param n Number of training examples for L2 regularization scaling
 */
void Network::display_backprop_gradients(const Eigen::VectorXd& x, const Eigen::VectorXd& y, size_t n) {
    auto [nabla_b, nabla_w] = backprop(x, y, n);
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== Bias Gradients (from backprop) ===" << std::endl;
    for (size_t i = 0; i < nabla_b.size(); ++i) {
        std::string layer_name = (i == nabla_b.size() - 1) ? "Output Layer" : "Hidden Layer " + std::to_string(i + 1);
        std::cout << layer_name << " (" << nabla_b[i].size() << " bias gradients):" << std::endl;
        for (int j = 0; j < nabla_b[i].size(); ++j) {
            std::cout << "  db[" << j << "] = " << nabla_b[i](j);
            if (j < nabla_b[i].size() - 1) std::cout << ",";
            if (j == 9 && nabla_b[i].size() > 10) {
                std::cout << " ... (truncated, total " << nabla_b[i].size() << " bias gradients)";
                break;
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    std::cout << "=== Weight Gradients (from backprop) ===" << std::endl;
    for (size_t i = 0; i < nabla_w.size(); ++i) {
        std::string from_layer = (i == 0) ? "Input Layer" : "Hidden Layer " + std::to_string(i);
        std::string to_layer = (i == nabla_w.size() - 1) ? "Output Layer" : "Hidden Layer " + std::to_string(i + 1);
        std::cout << "From " << from_layer << " to " << to_layer
            << " (" << nabla_w[i].rows() << "x" << nabla_w[i].cols() << " gradient matrix):" << std::endl;
        int max_rows = std::min(static_cast<int>(nabla_w[i].rows()), 5);
        int max_cols = std::min(static_cast<int>(nabla_w[i].cols()), 5);
        for (int r = 0; r < max_rows; ++r) {
            std::cout << "  [";
            for (int c = 0; c < max_cols; ++c) {
                std::cout << nabla_w[i](r, c);
                if (c < max_cols - 1) std::cout << ", ";
            }
            if (max_cols < nabla_w[i].cols()) std::cout << ", ...";
            std::cout << "]" << std::endl;
        }
        if (max_rows < nabla_w[i].rows() || max_cols < nabla_w[i].cols()) {
            std::cout << "  (Truncated, full size: " << nabla_w[i].rows() << "x" << nabla_w[i].cols() << ")" << std::endl;
        }
        std::cout << std::endl;
    }
}

/**
 * @brief Sets the weights of a specific layer.
 * @param layer_idx Index of the layer
 * @param weights New weight matrix
 */
void Network::set_layer_weights(size_t layer_idx, const Eigen::MatrixXd& weights) {
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
void Network::set_layer_biases(size_t layer_idx, const Eigen::VectorXd& biases) {
    if (layer_idx >= layers.size()) {
        throw std::out_of_range("Layer index out of bounds");
    }
    if (biases.size() != layers[layer_idx]->get_num_neurons()) {
        throw std::invalid_argument("Bias vector dimension mismatch");
    }
    layers[layer_idx]->set_biases(biases);
}


std::vector<double*> Network::createDeviceVectors(const std::vector<Eigen::VectorXd>& vec) {

    std::vector<double*> devicePointerStorage;

    for (size_t i = 0; i < vec.size(); ++i) {
        int size = vec[i].size();
        double* d_vec = nullptr;
        contextGPU_->allocate_biases(&d_vec, size);
        contextGPU_->copy_biases_to_device(d_vec, vec[i]);
        devicePointerStorage.push_back(d_vec);
    }

    return devicePointerStorage;
}


std::vector<double*> Network::createDeviceMatrices(const std::vector<Eigen::MatrixXd>& mat) {
    std::vector<double*> devicePointerStorage;

    for (size_t i = 0; i < mat.size(); ++i) {

        int m = mat[i].rows();
        int n = mat[i].cols();

        double* d_mat = nullptr;
        contextGPU_->allocate_weights(&d_mat, m, n);
        contextGPU_->copy_weights_to_device(d_mat, mat[i]);
        devicePointerStorage.push_back(d_mat);
    }

    return devicePointerStorage;
}


void Network::freeDevicePointers(std::vector<double*>& d_pointers) {

    for (auto& d_mem : d_pointers) {
        contextGPU_->free_vector(d_mem);
    }
}

bool Network::is_correct_prediction(const Eigen::VectorXd& output, int label) {
    Eigen::Index predicted;
    output.maxCoeff(&predicted);
    return predicted == static_cast<Eigen::Index>(label);
}

bool Network::is_correct_prediction(const Eigen::VectorXd& output, const Eigen::VectorXd& target) {
    Eigen::Index predicted, actual;
    output.maxCoeff(&predicted);
    target.maxCoeff(&actual);
    return predicted == actual;
}


// Example new helper
std::vector<double*> Network::get_layer_d_weight_grads() {
    std::vector<double*> grads;
    for (auto& layer : layers) {
        grads.push_back(layer->get_d_weight_grads_());
    }
    return grads;
}

std::vector<double*> Network::get_layer_d_delta() {
    std::vector<double*> deltas;
    for (auto& layer : layers) {
        deltas.push_back(layer->get_d_delta_());
    }
    return deltas;
}