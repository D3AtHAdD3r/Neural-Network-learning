#pragma once
#include"Network.hpp"
#include "CPUComputationContext.hpp"
#include "GPUComputationContext.hpp"
//#include "SigmoidActivation.hpp"
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

private:
    
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

    /**
     * @brief Computes activation for a vector of pre-activations.
     * @param z Pre-activation values
     * @return Activated values
     */
    Eigen::VectorXd computeActivation(const Eigen::VectorXd& z) const;

    /**
     * @brief Computes activation derivative for a vector of pre-activations.
     * @param z Pre-activation values
     * @return Derivative values
     */
    Eigen::VectorXd computeActivationDerivative(const Eigen::VectorXd& z) const;

public:
    /**
     * @brief Tests Layer constructor for correct initialization.
     */
    bool testLayerConstructor();

    /**
     * @brief Tests Layer forward pass for correct output and activation storage.
     */
    bool testLayerForward();

    /**
     * @brief Tests Layer gradient computation.
     */
    bool testLayerGradients();

    /**
     * @brief Tests Layer parameter updates.
     */
    bool testLayerUpdateParameters();


    bool testLayerComputeActivationDerivative();

public:
    //new network tests
    void testUpdateMiniBatchSimplified();

public:
    /**
     * @brief Tests Network constructor.
     */
    void testNetworkConstructor();

    /**
     * @brief Tests Network feedforward.
     */
    void testNetworkFeedforward();

    /**
     * @brief Tests Network backpropagation.
     */
    void testNetworkBackprop();

    /**
     * @brief Tests Network SGD on an XOR-like dataset.
     */
    void testNetworkSGD();

    /**
     * @brief Tests Network gradient checking to verify L2 regularization.
     */
    void testNetworkGradientChecking();


    // New Test for CPU vs GPU Consistency
    void testComputationContexts();

public:
    /**
     * @brief Runs all tests and reports results.
     * @return True if all tests passed, false otherwise
     */
    bool runAllTests();

private:
    /**
     * @brief Generates an XOR-like dataset for testing.
     * @param training_data Output vector for training data (input, target pairs)
     * @param test_data Output vector for test data (input, label pairs)
     */
    void generateXORLikeDataset(std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>& training_data,
        std::vector<std::pair<Eigen::VectorXd, int>>& test_data);

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

