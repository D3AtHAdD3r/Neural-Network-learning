#pragma once
#ifndef NETWORK_H
#define NETWORK_H

#include "Layer.hpp"
#include <Eigen/Dense>
#include <vector>
#include <random>
#include <algorithm>
#include <iostream>

/**
 * @brief A feedforward neural network with sigmoid activation.
 *
 * Implements a multi-layer neural network for tasks like MNIST classification,
 * supporting feedforward, backpropagation, and stochastic gradient descent (SGD).
 */
class Network {
public:
    /**
     * @brief Constructs a network with specified layer sizes.
     * @param sizes Vector of layer sizes (e.g., {784, 30, 10} for MNIST)
     * @param lambda L2 regularization parameter (default: 0.0, no regularization)
     */
    Network(const std::vector<int>& sizes, double lambda = 0.0);

    /**
     * @brief Computes the network output for a given input.
     * @param a Input vector
     * @return Output activations of the final layer
     */
    Eigen::VectorXd feedforward(const Eigen::VectorXd& a);

    /**
     * @brief Trains the network using stochastic gradient descent.
     * @param training_data Vector of (input, target) pairs
     * @param epochs Number of training epochs
     * @param mini_batch_size Size of each mini-batch
     * @param eta Learning rate
     * @param test_data Optional test data for evaluation
     * @param verbose If true, display detailed metrics per epoch
     */
    void SGD(std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>& training_data,
        int epochs, int mini_batch_size, double eta,
        const std::vector<std::pair<Eigen::VectorXd, int>>* test_data = nullptr,
        bool verbose = true);

    /**
     * @brief Displays biases for all layers.
     */
    void display_biases() const;

    /**
     * @brief Displays weights for all layers.
     */
    void display_weights() const;

    /**
     * @brief Displays biases layer-wise with truncation.
     * @param max_elements Maximum elements to display per layer
     */
    void display_layer_biases(int max_elements = 10) const;

    /**
     * @brief Displays weights layer-wise with truncation.
     * @param max_elements Maximum elements to display per layer
     */
    void display_layer_weights(int max_elements = 10) const;

    /**
     * @brief Displays gradients computed by backpropagation for a single example.
     * @param x Input vector
     * @param y Target vector
     */
    void display_backprop_gradients(const Eigen::VectorXd& x, const Eigen::VectorXd& y);

public:
    //Temporarily public
    /**
     * @brief Computes gradients for a single training example using backpropagation.
     * @param x Input vector
     * @param y Target vector
     * @return Pair of bias gradients and weight gradients for each layer
     */
    std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::MatrixXd>> backprop(
        const Eigen::VectorXd& x, const Eigen::VectorXd& y);

    /**
     * @brief Evaluates the network on test data and computes loss.
     * @param test_data Vector of (input, label) pairs
     * @return Pair of (correct predictions, total MSE loss)
     */
    std::pair<int, double> evaluate(const std::vector<std::pair<Eigen::VectorXd, int>>& test_data);

private:
    /**
     * @brief Updates weights and biases for a mini-batch and computes gradient norm.
     * @param mini_batch Vector of (input, target) pairs
     * @param eta Learning rate
     * @return L2 norm of gradients for the mini-batch
     */
    double update_mini_batch(const std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>& mini_batch, double eta);

    /**
     * @brief Computes the derivative of the cost function w.r.t. output activations.
     * @param output_activations Output activations of the final layer
     * @param y Target vector
     * @return Cost derivative
     */
    Eigen::VectorXd cost_derivative(const Eigen::VectorXd& output_activations, const Eigen::VectorXd& y) const;

private:
    /**
     * @brief Computes the mean squared error loss over test data.
     * @param test_data Vector of (input, label) pairs
     * @return Average MSE loss
     */
    double compute_test_loss(const std::vector<std::pair<Eigen::VectorXd, int>>& test_data);

    /**
     * @brief Computes the L2 norm of gradients for a mini-batch.
     * @param mini_batch Vector of (input, target) pairs
     * @return L2 norm of gradients
     */
    double compute_gradient_norm(const std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>& mini_batch);

private:
    int num_layers;                     ///< Number of layers
    std::vector<int> sizes;             ///< Sizes of each layer
    std::vector<Layer> layers;          ///< Layers of the network
    std::mt19937 rng;                   ///< Random number generator
    double last_test_loss;              ///< Cached test loss from evaluate
    double lambda;                      ///< L2 regularization parameter
};

/**
 * @brief Applies sigmoid activation element-wise to a vector.
 * @param z Input vector
 * @return Sigmoid of each element
 */
Eigen::VectorXd sigmoid(const Eigen::VectorXd& z);

/**
 * @brief Computes the derivative of the sigmoid function element-wise.
 * @param z Input vector
 * @return Sigmoid derivative for each element
 */
Eigen::VectorXd sigmoid_prime(const Eigen::VectorXd& z);

#endif