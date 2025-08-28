#pragma once
#ifndef NETWORK_H
#define NETWORK_H

#include "Layer.hpp"
#include "SigmoidActivation.hpp"
#include "ComputationContext.hpp"
#include <Eigen/Dense>
#include <vector>
#include <random>
#include <algorithm>
#include <iostream>
#include <memory>  

/**
 * @brief A feedforward neural network with sigmoid activation.
 *
 * Implements a multi-layer neural network for tasks like MNIST classification,
 * supporting feedforward, backpropagation, and stochastic gradient descent (SGD).
 * Supports both MSE and Cross-Entropy loss functions as well as L2 implementation.
 */
class Network {
public:
    enum class LossType { MSE, CROSS_ENTROPY };  // New enum for loss selection
    enum class NeuronType { SIGMOID };  // Start with only sigmoid
    
public:
    Network(
        const std::vector<int>& sizes, 
        double lambda = 0.0, 
        LossType loss_type = LossType::MSE, 
        NeuronType neuron_type = NeuronType::SIGMOID, 
        ComputationContext* context = nullptr, 
        unsigned int seed = std::random_device{}());

    ~Network(); 

public:
    Eigen::VectorXd feedforward(const Eigen::VectorXd& a);

    void SGD(std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>& training_data,
        int epochs, int mini_batch_size, double eta,
        const std::vector<std::pair<Eigen::VectorXd, int>>* test_data = nullptr,
        bool verbose = true);

    void display_biases() const;
    void display_weights() const;
    void display_layer_biases(int max_elements = 10) const;
    void display_layer_weights(int max_elements = 10) const;
    void display_backprop_gradients(const Eigen::VectorXd& x, const Eigen::VectorXd& y, size_t n);

public:
    //setters
    void set_layer_weights(size_t layer_idx, const Eigen::MatrixXd& weights);
    void set_layer_biases(size_t layer_idx, const Eigen::VectorXd& biases);
    
public:
    //getters
    const std::vector<std::unique_ptr<Layer>>& get_layers() const { return layers; }
 
    //For unit tests(NeuralNetworkTest). TODO: make them private or protected and expose it through friend test classes
    std::vector<std::unique_ptr<Layer>>& get_mutable_layers() { return layers; }

    //helper to get per-layer d_grads
    std::vector<double*> get_layer_d_weight_grads();
    std::vector<double*> get_layer_d_bias_grads();

public:
    //Temporarily public
    // CPU-specific backprop (returns host gradients)
    std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::MatrixXd>> backprop_cpu(
        const Eigen::VectorXd& x, const Eigen::VectorXd& y, size_t n);

    // GPU-specific backprop (accumulates gradients directly on device; returns empty for compatibility)
    std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::MatrixXd>> backprop_gpu(
        const Eigen::VectorXd& x, const Eigen::VectorXd& y, size_t n);

    // Wrapper to dispatch based on context
    std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::MatrixXd>> backprop(
        const Eigen::VectorXd& x, const Eigen::VectorXd& y, size_t n);

    std::pair<int, double> evaluate(const std::vector<std::pair<Eigen::VectorXd, int>>& test_data, size_t n);

    // New overload for training data (target as one-hot vector)
    std::pair<int, double> evaluate(const std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>& test_data, size_t n);

    double update_mini_batch(const std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>& mini_batch, double eta, size_t n);

private:
    Eigen::VectorXd cost_derivative(const Eigen::VectorXd& output_activations, const Eigen::VectorXd& y) const;

private:
    double compute_test_loss(const std::vector<std::pair<Eigen::VectorXd, int>>& test_data);
    double compute_gradient_norm(const std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>& mini_batch, size_t n);

private:
    //Temporary helpers for cuda operations
    
    //Returns dynamic created pointers, cleanup is upto callers
    std::vector<double*> createDeviceVectors(const std::vector<Eigen::VectorXd>& vec);
    std::vector<double*> createDeviceMatrices(const std::vector<Eigen::MatrixXd>& mat);

    void freeDevicePointers(std::vector<double*>& d_pointers);

private:
    //Helpers
    bool is_correct_prediction(const Eigen::VectorXd& output, int label);
    bool is_correct_prediction(const Eigen::VectorXd& output, const Eigen::VectorXd& target);

private:
    int num_layers;                                 ///< Number of layers
    std::vector<int> sizes;                         ///< Sizes of each layer
    std::vector<std::unique_ptr<Layer>> layers;     ///< Layers of the network
    std::mt19937 rng;                               ///< Random number generator
    double last_test_loss;                          ///< Cached test loss from evaluate
    double lambda;                                  ///< L2 regularization parameter
    LossType loss_type_;                            ///< Type of loss function to use
    NeuronType neuron_type_;                        ///< Track chosen neuron type
    std::unique_ptr<Activation> activation_;        ///< Dynamic activation instance
    ComputationContext* context_;                   // Raw pointer
    GPUComputationContext* contextGPU_ = nullptr;
    CPUComputationContext* contextCPU_ = nullptr;
    bool owns_context_;                             // Flag to track ownership
    bool is_gpu_context_;                           // New: Cached flag for quick dispatch

private:
    //GPU storage pointers
    std::vector<double*> accumulate_weight_grads;
    std::vector<double*> accumulate_bias_grads;
    //storage dimensions
    std::vector<int> weight_rows;
    std::vector<int> weight_cols;
    std::vector<int> bias_sizes;
};

#endif