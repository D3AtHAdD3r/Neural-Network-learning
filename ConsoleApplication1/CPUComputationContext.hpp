#include "ComputationContext.hpp"




// CPU implementation of the ComputationContext using Eigen library
class CPUComputationContext : public ComputationContext {
public:
    // Compute linear transformation: weights * input + biases
    Eigen::VectorXd computeLinear(const Eigen::MatrixXd& weights,
        const Eigen::VectorXd& input,
        const Eigen::VectorXd& biases) override;

    // Apply activation function to the pre-activation values
    Eigen::VectorXd applyActivation(const Eigen::VectorXd& z,
        const Activation* activation) override;

    // Compute the derivative of the activation function
    Eigen::VectorXd computeActivationDerivative(const Eigen::VectorXd& activations,
        const Eigen::VectorXd& pre_activations,
        const Activation* activation) override;

    // Compute gradients for weights and biases
    void computeGradients(const Eigen::VectorXd& deltas,
        const Eigen::VectorXd& activation_derives,
        const Eigen::VectorXd& input,
        Eigen::MatrixXd& weight_grads,
        Eigen::VectorXd& bias_grads) override;
   
    // Update parameters by subtracting gradients
    void updateParameters(Eigen::MatrixXd& weights,
        Eigen::VectorXd& biases,
        const Eigen::MatrixXd& weight_grads,
        const Eigen::VectorXd& bias_grads, double scale) override;
};

// Optional: Factory function to create an instance (not required but useful)
// CPUComputationContext* createCPUContext() {
//     return new CPUComputationContext();
// }