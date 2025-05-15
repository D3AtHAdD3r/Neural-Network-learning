#pragma once
#ifndef NETWORK_H
#define NETWORK_H

#include <Eigen/Dense>
#include <vector>
#include <random>
#include <algorithm>
#include <iostream>


class Network {
public:
    Network(const std::vector<int>& sizes);
    Eigen::VectorXd feedforward(const Eigen::VectorXd& a); //Computes the network’s output for an input vector.
    void SGD(std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>& training_data,
        int epochs, int mini_batch_size, double eta,
        const std::vector<std::pair<Eigen::VectorXd, int>>* test_data = nullptr);
public:
    //Debug functions
    void display_biases() const;  // New function to display biases
    void display_weights() const;  // New function to display weights
    void display_backprop_gradients(const Eigen::VectorXd& x, const Eigen::VectorXd& y) const; // New function
public:
    //Temporarily made public
    std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::MatrixXd>> backprop(
        const Eigen::VectorXd& x, const Eigen::VectorXd& y) const;
public:
    //For clarity, functions
    void feedforward_standalone(const Eigen::VectorXd& x);

private:
    void update_mini_batch(const std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>& mini_batch, double eta);
    /*std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::MatrixXd>> backprop(
        const Eigen::VectorXd& x, const Eigen::VectorXd& y) const;*/
    int evaluate(const std::vector<std::pair<Eigen::VectorXd, int>>& test_data);
    Eigen::VectorXd cost_derivative(const Eigen::VectorXd& output_activations, const Eigen::VectorXd& y) const;

    int num_layers;
    std::vector<int> sizes;
    std::vector<Eigen::VectorXd> biases;
    std::vector<Eigen::MatrixXd> weights;
    std::mt19937 rng; // Random number generator
};

Eigen::VectorXd sigmoid(const Eigen::VectorXd& z);
Eigen::VectorXd sigmoid_prime(const Eigen::VectorXd& z);

#endif