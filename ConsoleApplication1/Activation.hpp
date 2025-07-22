#pragma once

#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

#include <Eigen/Dense>
#include <cudnn.h>

/**
 * @brief Abstract base class for activation functions.
 *
 * Defines the interface for activation functions used in neural network layers.
 * Subclasses must implement activate and derivative methods.
 * Note: Virtual function calls incur vtable lookup overhead. 
 * Consider function pointers for performance-critical applications after profiling.
 */

class Activation {
public:
	virtual ~Activation() = default;  // Virtual destructor for proper cleanup

    /**
     * @brief Applies the activation function to the input.
     * @param z Input vector (pre-activations)
     * @return Activated output vector
     */
    virtual Eigen::VectorXd activate(const Eigen::VectorXd& z) const = 0;

    /**
     * @brief Computes the derivative of the activation function.
     * @param activations Activated output vector (optional, use if needed)
     * @param pre_activations Pre-activation vector (optional, use if needed)
     * @return Derivative vector
     */
    virtual Eigen::VectorXd derivative(const Eigen::VectorXd* activations = nullptr, const Eigen::VectorXd* pre_activations = nullptr) const = 0;

    /**
     * @brief Returns the cuDNN activation mode for GPU computation.
     * @return cuDNN activation mode
     * @throws std::runtime_error if the activation is not supported by cuDNN
     */
    virtual cudnnActivationMode_t getCudnnActivationMode() const = 0;
};



#endif // ACTIVATION_HPP