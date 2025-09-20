#pragma once
#ifndef NETWORK_B_H
#define NETWORK_B_H

#include"Layer_b.hpp"
#include <Eigen/Dense>
#include <vector>
#include <random>
#include <memory> 

class Network_b {
public:
    enum class LossType { MSE, CROSS_ENTROPY };  // New enum for loss selection
    enum class NeuronType { SIGMOID };  // Start with only sigmoid

public:
    Network_b(
        const std::vector<int>& sizes,
        double lambda = 0.0,
        LossType loss_type = LossType::MSE,
        NeuronType neuron_type = NeuronType::SIGMOID,
        unsigned int seed = std::random_device{}(),
        int max_batch_size = MAX_BATCH_SIZE);

    ~Network_b();

public:
    void init_batch_buffers(int mini_batch_size);

    void SGD(std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>& training_data,
        int epochs, int mini_batch_size, double eta,
        const std::vector<std::pair<Eigen::VectorXd, int>>* test_data = nullptr,
        bool verbose = true);

    // Batched feedforward on GPU: Processes a batch of inputs
    // batch_inputs: Vector of input vectors (size <= allocated_batch_size_)
    // batch_outputs: Output vector to store results (resized to batch_inputs.size())
    void feedforward_batch(const std::vector<Eigen::VectorXd>& batch_inputs, std::vector<Eigen::VectorXd>& batch_outputs);

    // Batched backprop: Assumes feedforward_gpu_batch was run; accumulates grads, returns avg loss
    void backprop_batch(const std::vector<Eigen::VectorXd>& batch_targets, double& batch_loss);

    double update_mini_batch(const std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>& mini_batch, double eta, size_t n);

    std::pair<int, double> evaluate_batch(const std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>& test_data, size_t n);
    std::pair<int, double> evaluate_batch(const std::vector<std::pair<Eigen::VectorXd, int>>& test_data, size_t n);
public:
    //setters
    void set_layer_weights(size_t layer_idx, const Eigen::MatrixXd& weights);
    void set_layer_biases(size_t layer_idx, const Eigen::VectorXd& biases);
public:
    //getters
    const std::vector<std::unique_ptr<Layer_b>>& get_layers() const { return layers; }
    std::vector<std::unique_ptr<Layer_b>>& get_mutable_layers() { return layers; }
    int get_num_layers() { return num_layers; };
    const std::vector<int>& get_layer_sizes() const { return sizes; };
    std::vector<double*> get_accumulate_weight_grads() const { return accumulate_weight_grads; }
    std::vector<double*> get_accumulate_bias_grads() const { return accumulate_bias_grads; }

private:
    //Helpers
    bool is_correct_prediction(const Eigen::VectorXd& output, int label);
    bool is_correct_prediction(const Eigen::VectorXd& output, const Eigen::VectorXd& target);
    Eigen::VectorXd label_to_one_hot(int label) const;

private:
    static const int MAX_BATCH_SIZE = 128;
    int num_layers;                                 ///< Number of layers
    std::vector<int> sizes;                         ///< Sizes of each layer
    std::mt19937 rng;                               ///< Random number generator
    double last_test_loss;                          ///< Cached test loss from evaluate
    double lambda;                                  ///< L2 regularization parameter
    LossType loss_type_;                            ///< Type of loss function to use
    NeuronType neuron_type_;                        ///< Track chosen neuron type
    std::vector<std::unique_ptr<Layer_b>> layers;     ///< Layers of the network
    std::unique_ptr<Activation> activation_;        ///< Dynamic activation instance
    std::unique_ptr<GPUPass> gpuCtx_;
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
    double* d_input_main = nullptr;             //??
private:
    int max_batch_size_;
    int allocated_batch_size_;                  // Actual size allocated for batch buffers
    bool batch_buffers_allocated_ = false;      // Tracks if batch buffers are allocated
private:
    // Batch-related GPU buffers (centralized here for efficiency; avoids per-Layer allocation overhead)
    double* d_batch_main_input = nullptr;       // GPU buffer for main batch input (input_size * allocated_batch_size_)
    std::vector<double*> d_batch_pre_activations;  // Per-layer pre-activations (neurons * allocated_batch_size_)
    std::vector<double*> d_batch_activations;      // Per-layer activations (neurons * allocated_batch_size_)
    std::vector<double*> d_batch_deltas;  // Per-layer batched deltas (neurons x allocated_batch_size_)
    double* d_batch_targets = nullptr;    // Batched targets (output_size x allocated_batch_size_)
    double* d_temp_deriv_batch = nullptr; // Temporary buffer for batched derivatives
};

#endif