#pragma once
#include <Eigen/Dense>

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
