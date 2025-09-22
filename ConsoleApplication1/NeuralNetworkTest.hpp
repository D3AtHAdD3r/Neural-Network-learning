#pragma once
#include"Network.hpp"
#include <Eigen/Dense>
#include <vector>
#include <string>

/**
 * @brief Unit test class for neural network components.
 *
 * Encapsulates tests for Layer and Network classes, with configurable parameters
 * for layer and network sizes. Designed to be activation-function agnostic.
 */
class NeuralNetworkTest
{
public:
    /**
     * @brief Constructs a test suite with configurable parameters.
     * @param layer_inputs Number of inputs for Layer tests (default: 2)
     * @param layer_neurons Number of neurons for Layer tests (default: 3)
     * @param seed Random seed for initialization (default: 42)
     * @param network_sizes Network layer sizes (default: {2, 3, 2})
     */
    NeuralNetworkTest(int layer_inputs = 2, int layer_neurons = 3, unsigned int seed = 42,
        const std::vector<int>& network_sizes = { 2, 3, 2 }, Network::NeuronType neuron_type = Network::NeuronType::SIGMOID);

    ~NeuralNetworkTest();
public:
    //non-Batched tests
    bool testNetworkBackprop();
    void runUpdateMiniBatchTests(const std::string& context, const std::string& loss, double lambda, int batch_size);
    bool testUpdateMiniBatch();
public:
    //Batched func tests
    bool test_feedforward_batch_vs_single(int batch_size, int input_size, int hidden_size, int output_size, GPUComputationContext* gpu_context);
    bool test_feedforward_gpu_batch();
    bool test_backprop_gpu_batch();
    bool test_update_mini_batch_batch();
public:
    //Batched gpu computation context func tests
    bool testBatchFunctionsGPU_context();
    bool test_launch_elementwise_subtract_batch();
    bool test_launch_elementwise_multiply_batch();
    bool test_computeGradientsGPU_batch();
    bool test_compute_delta_back_batch();
    bool test_computeActivationDerivativeGPU_batch();
    bool test_cost_prime_mse_crossent_batched();
    bool test_compute_mse_loss_batch_gpu();
    bool test_evaluate_batch();
    bool test_evaluate_batch_2();
public:
    bool runAllTests();
public:
    bool customtest();
public:
    /**
     * @brief Generates an XOR-like dataset for testing.
     * @param training_data Output vector for training data (input, target pairs)
     * @param test_data Output vector for test data (input, label pairs)
     */
    void generateXORLikeDataset(std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>& training_data,
        std::vector<std::pair<Eigen::VectorXd, int>>& test_data);

    Eigen::VectorXd generate_random_input(int size, unsigned int seed);
private:

    /**
     * @brief Provides precomputed test data for backpropagation verification.
     * @param loss_type The loss function type (MSE or CROSS_ENTROPY)
     * @param weights Output: Predefined weights for each layer
     * @param biases Output: Predefined biases for each layer
     * @param x Output: Input vector
     * @param y Output: Target vector
     * @param expected_nabla_b Output: Expected bias gradients
     * @param expected_nabla_w Output: Expected weight gradients
     * @note neuron support: sigmoid only, no regularization
     */
    void getPrecomputedBackpropTestData(Network::LossType loss_type,
        std::vector<Eigen::MatrixXd>& weights,
        std::vector<Eigen::VectorXd>& biases,
        Eigen::VectorXd& x,
        Eigen::VectorXd& y,
        std::vector<Eigen::VectorXd>& expected_nabla_b,
        std::vector<Eigen::MatrixXd>& expected_nabla_w) const;
    
    /**
     * @brief Asserts a condition is true, reporting failure if not.
     * @param cond Condition to check
     * @param message Error message to display on failure
     * @param file Source file name
     * @param line Line number
     */
    void assertTrue(bool cond, const std::string& message, const char* file, int line);


    /**
     * @brief Asserts two doubles are approximately equal within a tolerance.
     * @param a First value
     * @param b Second value
     * @param tol Tolerance
     * @param message Error message
     * @param file Source file name
     * @param line Line number
     */
    void assertApprox(double a, double b, double tol, const std::string& message, const char* file, int line);
    void assertVectorApprox(const Eigen::VectorXd& a, const Eigen::VectorXd& b, double tol, const std::string& message, const char* file, int line);
    void assertMatrixApprox(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b, double tol, const std::string& message, const char* file, int line);

private:
    
private:
    static constexpr double TOL = 1e-3; ///< Tolerance for floating-point comparisons
private:
    int layer_inputs_;                  ///< Number of inputs for Layer tests
    int layer_neurons_;                 ///< Number of neurons for Layer tests
    unsigned int seed_;                 ///< Random seed for reproducibility
    std::vector<int> network_sizes_;    ///< Network layer sizes
    int passed_tests_;                  ///< Count of passed tests
    int total_tests_;                   ///< Total tests run
    Network::NeuronType neuron_type_;            ///< Track chosen neuron type
    std::unique_ptr<Activation> activation_;  ///< Dynamic activation instance
private:
    std::unique_ptr<CPUComputationContext> cpuContext; ///< CPU computation context
    std::unique_ptr<GPUComputationContext> gpuContext; ///< GPU computation context
    std::vector<ComputationContext*> computeContexts; ///< List of computation contexts
};

