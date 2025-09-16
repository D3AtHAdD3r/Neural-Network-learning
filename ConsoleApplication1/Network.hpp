#pragma once
#ifndef NETWORK_H
#define NETWORK_H

#include "Layer.hpp"
#include "SigmoidActivation.hpp"
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
        unsigned int seed = std::random_device{}(), 
        int max_batch_size = GPUComputationContext::MAX_BATCH_SIZE);

    ~Network(); 

public:
    //Eigen::VectorXd feedforward(const Eigen::VectorXd& a);
    Eigen::VectorXd feedforward_cpu(const Eigen::VectorXd& a);
    Eigen::VectorXd feedforward_gpu(const Eigen::VectorXd& a);

    void SGD(std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>& training_data,
        int epochs, int mini_batch_size, double eta,
        const std::vector<std::pair<Eigen::VectorXd, int>>* test_data = nullptr,
        bool verbose = true);

    double update_mini_batch(const std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>& mini_batch, double eta, size_t n);

    std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::MatrixXd>> backprop_cpu(
        const Eigen::VectorXd& x, const Eigen::VectorXd& y, size_t n);

    // GPU-specific backprop (accumulates gradients directly on device; returns empty for compatibility)
    std::pair<std::vector<double*>, std::vector<double*>> backprop_gpu(
        const Eigen::VectorXd& x, const Eigen::VectorXd& y, size_t n);

    std::pair<int, double> evaluate(const std::vector<std::pair<Eigen::VectorXd, int>>& test_data, size_t n);
    std::pair<int, double> evaluate(const std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>& test_data, size_t n);

public:
    //Batched funcs
    // Batched feedforward on GPU: Processes a batch of inputs
    // batch_inputs: Vector of input vectors (size <= allocated_batch_size_)
    // batch_outputs: Output vector to store results (resized to batch_inputs.size())
    void feedforward_gpu_batch(const std::vector<Eigen::VectorXd>& batch_inputs, std::vector<Eigen::VectorXd>& batch_outputs);
    double update_mini_batch_batch(const std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>& mini_batch, double eta, size_t n);
    void init_batch_buffers(int mini_batch_size);
public:
    //Debug
    void display_biases() const;
    void display_weights() const;
    void display_layer_biases(int max_elements = 10) const;
    void display_layer_weights(int max_elements = 10) const;
  
public:
    void set_layer_weights(size_t layer_idx, const Eigen::MatrixXd& weights);
    void set_layer_biases(size_t layer_idx, const Eigen::VectorXd& biases);
    const std::vector<std::unique_ptr<Layer>>& get_layers() const { return layers; }
    std::vector<std::unique_ptr<Layer>>& get_mutable_layers() { return layers; } //For unit tests(NeuralNetworkTest). 
    std::vector<double*> get_layer_d_weight_grads();
    std::vector<double*> get_layer_d_delta(); //d_delta = d_bias_grads
    int get_num_layers() { return num_layers; };
    const std::vector<int>& get_layer_sizes() const { return sizes; };
private:
    Eigen::VectorXd cost_derivative(const Eigen::VectorXd& output_activations, const Eigen::VectorXd& y) const;

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
    // Temporary buffers for scaled gradients
    std::vector<double*> temp_weight_grads;         
    std::vector<double*> temp_bias_grads;          
    //Device Buffer for main input(layer1) 
    double* d_input_main = nullptr;

 private:
    int max_batch_size_;
    int allocated_batch_size_;                  // Actual size allocated for batch buffers
    bool batch_buffers_allocated_ = false;      // Tracks if batch buffers are allocated
    // Batch-related GPU buffers (centralized here for efficiency; avoids per-Layer allocation overhead)
    double* d_batch_main_input = nullptr;       // GPU buffer for main batch input (input_size * allocated_batch_size_)
    std::vector<double*> d_batch_pre_activations;  // Per-layer pre-activations (neurons * allocated_batch_size_)
    std::vector<double*> d_batch_activations;      // Per-layer activations (neurons * allocated_batch_size_)
};

#endif