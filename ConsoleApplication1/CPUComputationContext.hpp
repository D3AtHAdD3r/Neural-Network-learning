#include "ComputationContext.hpp"


// CPU implementation of the ComputationContext using Eigen library
class CPUComputationContext : public ComputationContext {

public:
    void fuckingHell() override;
public:
    // Compute linear transformation: weights * input + biases
    Eigen::VectorXd computeLinearCPU(const Eigen::MatrixXd& weights,
        const Eigen::VectorXd& input,
        const Eigen::VectorXd& biases);

    // New method for computing weight gradients (outer product)
    Eigen::MatrixXd computeWeightGradientCPU(const Eigen::VectorXd& delta,
        const Eigen::VectorXd& activation);

    // Apply activation function to the pre-activation values
    Eigen::VectorXd applyActivationCPU(const Eigen::VectorXd& z,
        const Activation* activation);

    // Compute the derivative of the activation function
    Eigen::VectorXd computeActivationDerivativeCPU(const Eigen::VectorXd& activations,
        const Eigen::VectorXd& pre_activations,
        const Activation* activation);

    // Compute gradients for weights and biases
    /*void computeGradientsCPU(const Eigen::VectorXd& deltas,
        const Eigen::VectorXd& activation_derives,
        const Eigen::VectorXd& input,
        Eigen::MatrixXd& weight_grads,
        Eigen::VectorXd& bias_grads);*/

    void computeGradientsCPU(
        const Eigen::VectorXd& deltas,
        const Eigen::VectorXd& activation_derives,
        const Eigen::VectorXd& input,
        Eigen::MatrixXd& weight_grads,
        Eigen::VectorXd& bias_grads,
        bool apply_derivative);
   
    // Update parameters by subtracting gradients
    void updateParametersCPU(Eigen::MatrixXd& weights,
        Eigen::VectorXd& biases,
        const Eigen::MatrixXd& weight_grads,
        const Eigen::VectorXd& bias_grads, double scale);

    void accumulateGradientsCPU(const std::vector<Eigen::MatrixXd>& weight_grads_in,
        const std::vector<Eigen::VectorXd>& bias_grads_in,
        std::vector<Eigen::MatrixXd>& weight_grads_out,
        std::vector<Eigen::VectorXd>& bias_grads_out,
        double scale);

public:
    double compute_squared_normCPU(const Eigen::MatrixXd& matrix);
    double compute_squared_normCPU(const Eigen::VectorXd& vector);
    double compute_mse_lossCPU(const Eigen::VectorXd& output, const Eigen::VectorXd& target);
    double compute_cross_entropy_lossCPU(const Eigen::VectorXd& output, const Eigen::VectorXd& target);
};

