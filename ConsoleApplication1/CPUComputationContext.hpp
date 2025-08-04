#include "ComputationContext.hpp"


// CPU implementation of the ComputationContext using Eigen library
class CPUComputationContext : public ComputationContext {
public:
    // Compute linear transformation: weights * input + biases
    Eigen::VectorXd computeLinear(const Eigen::MatrixXd& weights,
        const Eigen::VectorXd& input,
        const Eigen::VectorXd& biases) override;

    // New method for computing weight gradients (outer product)
    Eigen::MatrixXd computeWeightGradient(const Eigen::VectorXd& delta,
        const Eigen::VectorXd& activation) override;

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

    void accumulateGradients(const std::vector<Eigen::MatrixXd>& weight_grads_in,
        const std::vector<Eigen::VectorXd>& bias_grads_in,
        std::vector<Eigen::MatrixXd>& weight_grads_out,
        std::vector<Eigen::VectorXd>& bias_grads_out,
        double scale) override;

    double compute_squared_norm(const Eigen::MatrixXd& matrix) override;

    double compute_mse_loss(const Eigen::VectorXd& output, const Eigen::VectorXd& target) override;

    double compute_cross_entropy_loss(const Eigen::VectorXd& output, const Eigen::VectorXd& target) override;

    // Memory management for GPU
    void allocate_weights(double** d_weights, int rows, int cols) override;
    void allocate_biases(double** d_biases, int size) override;
    void copy_weights_to_device(double* d_weights, const Eigen::MatrixXd& weights) override;
    void copy_biases_to_device(double* d_biases, const Eigen::VectorXd& biases) override;
    void free_weights(double* d_weights) override;
    void free_biases(double* d_biases) override;

    // methods to support GPU memory allocation and operations.
    void allocate_vector(double** d_vector, int size) override;
    void free_vector(double* d_vector) override;
    void copy_to_device(double* d_vector, const Eigen::VectorXd& vector) override;
    void copy_to_host(Eigen::VectorXd& vector, double* d_vector, int size) override;
    void computeLinearGPU(double* d_weights, double* d_input, double* d_biases, double* d_z, int m, int n) override;
    void applyActivationGPU(double* d_z, double* d_a, int n, const Activation* activation) override;

    Eigen::VectorXd computeActivationDerivativeGPU(double* d_a, double* d_z, double* d_dy, double* d_derivatives, int size, const Activation* activation) override;
    void computeGradientsGPU(const Eigen::VectorXd& deltas, double* d_derivatives, double* d_input, Eigen::MatrixXd& weight_grads, Eigen::VectorXd& bias_grads, int m, int n) override;
    
    //debug
    void debugPrint(const double* data, int n) override;
};

// Optional: Factory function to create an instance (not required but useful)
// CPUComputationContext* createCPUContext() {
//     return new CPUComputationContext();
// }