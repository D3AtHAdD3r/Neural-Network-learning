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

    virtual double compute_squared_norm(const Eigen::MatrixXd& matrix) = 0;

    virtual double compute_mse_loss(const Eigen::VectorXd& output, const Eigen::VectorXd& target) = 0;

    virtual double compute_cross_entropy_loss(const Eigen::VectorXd& output, const Eigen::VectorXd& target) = 0;

    // Memory management for GPU
    virtual void allocate_weights(double** d_weights, int rows, int cols) = 0;
    virtual void allocate_biases(double** d_biases, int size) = 0;
    virtual void copy_weights_to_device(double* d_weights, const Eigen::MatrixXd& weights) = 0;
    virtual void copy_biases_to_device(double* d_biases, const Eigen::VectorXd& biases) = 0;
    virtual void copy_weights_to_host(Eigen::MatrixXd& weights, double* d_weights, int rows, int cols) = 0;
    virtual void copy_biases_to_host(Eigen::VectorXd& biases, double* d_biases, int size) = 0;
    virtual void free_weights(double* d_weights) = 0;
    virtual void free_biases(double* d_biases) = 0;

    //methods to support GPU memory allocation and operations.
    virtual void allocate_vector(double** d_vector, int size) = 0;
    virtual void free_vector(double* d_vector) = 0;
    virtual void copy_to_device(double* d_vector, const Eigen::VectorXd& vector) = 0;
    virtual void copy_to_host(Eigen::VectorXd& vector, double* d_vector, int size) = 0;
    virtual void copy_to_host(Eigen::MatrixXd& matrix, double* d_matrix, int rows, int cols) =0;
    virtual void computeLinearGPU(double* d_weights, double* d_input, double* d_biases, double* d_z, int m, int n) = 0;
    virtual void applyActivationGPU(double* d_z, double* d_a, int n, const Activation* activation) = 0;

    virtual Eigen::VectorXd computeActivationDerivativeGPU(double* d_a, double* d_z, double* d_dy, double* d_derivatives, int size, const Activation* activation) = 0;
    virtual void computeGradientsGPU(const Eigen::VectorXd& deltas, double* d_derivatives, double* d_input, Eigen::MatrixXd& weight_grads, Eigen::VectorXd& bias_grads, int m, int n) = 0;
    virtual void updateParametersGPU(double* d_weights,
            double* d_biases,
            const Eigen::MatrixXd& weight_grads,
            const Eigen::VectorXd& bias_grads,
            int m, int n, int bias_size, double scale) = 0;
    
    virtual void debugPrint(const double* data, int n) = 0;
}; 


#endif // COMPUTATION_CONTEXT_HPP
