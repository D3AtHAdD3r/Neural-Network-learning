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
};


#endif // COMPUTATION_CONTEXT_HPP
