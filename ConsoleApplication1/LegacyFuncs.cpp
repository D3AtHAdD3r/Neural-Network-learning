#include"LegacyFuncs.h"

/**
 * @brief Applies sigmoid activation element-wise to a vector.
 * @param z Input vector
 * @return Sigmoid of each element
 */
Eigen::VectorXd sigmoid(const Eigen::VectorXd& z) {
    return z.unaryExpr([](double x) { return 1.0 / (1.0 + std::exp(-x)); });
}

/**
 * @brief Computes the derivative of the sigmoid function element-wise.
 * @param z Input vector
 * @return Sigmoid derivative for each element
 */
Eigen::VectorXd sigmoid_prime(const Eigen::VectorXd& z) {
    Eigen::VectorXd sz = sigmoid(z);
    return sz.cwiseProduct(Eigen::VectorXd::Ones(sz.size()) - sz);
}