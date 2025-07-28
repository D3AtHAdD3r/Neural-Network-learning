#pragma once
#ifndef COMPUTATION_CONTEXT_HPP
#define COMPUTATION_CONTEXT_HPP

#include <Eigen/Dense>
#include "Activation.hpp"


// Abstract base class defining the computation interface for neural network layers
class ComputationContext {
public:
    // Virtual destructor to ensure proper cleanup in derived classes
    virtual ~ComputationContext() = default;

    // Compute the linear transformation: weights * input + biases
    virtual Eigen::VectorXd computeLinear(const Eigen::MatrixXd& weights,
        const Eigen::VectorXd& input,
        const Eigen::VectorXd& biases) = 0;

    // New method for computing weight gradients (outer product)
    virtual Eigen::MatrixXd computeWeightGradient(const Eigen::VectorXd& delta, 
        const Eigen::VectorXd& activation) = 0;

    // Apply the activation function to the pre-activation values
    virtual Eigen::VectorXd applyActivation(const Eigen::VectorXd& z,
        const Activation* activation) = 0;

    // Compute the derivative of the activation function
    virtual Eigen::VectorXd computeActivationDerivative(const Eigen::VectorXd& activations,
        const Eigen::VectorXd& pre_activations,
        const Activation* activation) = 0;

    // Compute gradients for weights and biases based on backpropagated deltas
    virtual void computeGradients(const Eigen::VectorXd& deltas,
        const Eigen::VectorXd& activation_derives,
        const Eigen::VectorXd& input,
        Eigen::MatrixXd& weight_grads,
        Eigen::VectorXd& bias_grads) = 0;

    // Update weights and biases using the computed gradients
    virtual void updateParameters(Eigen::MatrixXd& weights,
        Eigen::VectorXd& biases,
        const Eigen::MatrixXd& weight_grads,
        const Eigen::VectorXd& bias_grads, double scale) = 0;

    // New method for GPU gradient accumulation
    // Accumulates gradients into weight_grads and bias_grads: weight_grads += alpha * delta_nabla_w.
    // Typically, alpha = 1.0 to sum raw gradients; scaling by eta / mini_batch.size() is handled in update_parameters.
    virtual void accumulateGradients(const std::vector<Eigen::MatrixXd>& weight_grads_in,
        const std::vector<Eigen::VectorXd>& bias_grads_in,
        std::vector<Eigen::MatrixXd>& weight_grads_out,
        std::vector<Eigen::VectorXd>& bias_grads_out,
        double scale) = 0;
};


#endif // COMPUTATION_CONTEXT_HPP
