#pragma once
#ifndef SIGMOID_ACTIVATION_HPP
#define SIGMOID_ACTIVATION_HPP

#include "Activation.hpp"
#include <cmath>

/**
 * @brief Implementation of the sigmoid activation function.
 */
class SigmoidActivation : public Activation {
public:
    Eigen::VectorXd activate(const Eigen::VectorXd& z) const override {
        return z.unaryExpr([](double x) { return 1.0 / (1.0 + std::exp(-x)); });
    }

    Eigen::VectorXd derivative(const Eigen::VectorXd* activations = nullptr, const Eigen::VectorXd* pre_activations = nullptr) const override {
        if (activations && !activations->size()) {
            throw std::runtime_error("Invalid activations vector size");
        }
        const Eigen::VectorXd& a = activations ? *activations : activate(*pre_activations);  // Use activations if provided, else compute from pre-activations
        return a.cwiseProduct(Eigen::VectorXd::Ones(a.size()) - a);
    }

    cudnnActivationMode_t getCudnnActivationMode() const override {
        return CUDNN_ACTIVATION_SIGMOID;
    }
};


#endif // SIGMOID_ACTIVATION_HPP