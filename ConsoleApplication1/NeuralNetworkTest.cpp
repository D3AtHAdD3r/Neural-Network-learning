#include "NeuralNetworkTest.hpp"
#include"Network.hpp"
#include <iostream>
#include <cmath>
#include <cassert>


NeuralNetworkTest::NeuralNetworkTest(int layer_inputs, int layer_neurons, unsigned int seed, const std::vector<int>& network_sizes) :
    layer_inputs_(layer_inputs), layer_neurons_(layer_neurons), seed_(seed),
    network_sizes_(network_sizes), passed_tests_(0), total_tests_(0)
{}


void NeuralNetworkTest::assertTrue(bool cond, const std::string& message, const char* file, int line)
{
    if (!cond) {
        std::cerr << "Assertion failed: " << message << " at " << file << ":" << line << std::endl;
        exit(1);
    }
}

void NeuralNetworkTest::assertApprox(double a, double b, double tol, const std::string& message, const char* file, int line)
{
    if (std::abs(a - b) > tol) {
        std::cerr << "Assertion failed: " << a << " != " << b << " (" << message << ") at " << file << ":" << line << std::endl;
        exit(1);
    }
}

void NeuralNetworkTest::testLayerConstructor()
{
    std::cout << "Running testLayerConstructor... ";
    ++total_tests_;
    Layer layer(layer_inputs_, layer_neurons_, seed_);
    const auto& weights = layer.get_weights();
    const auto& biases = layer.get_biases();

    //Check Layer's Weight-Matrix size
    assertTrue(weights.rows() == layer_neurons_ && weights.cols() == layer_inputs_,
        "Incorrect weight matrix size", __FILE__, __LINE__);

    //Check Layer's Bias-Vector size
    assertTrue(biases.size() == layer_neurons_, "Incorrect bias vector size", __FILE__, __LINE__);

    //Checks Weight-Matrix's and Bias-Vector's initialization values
    double stddev = std::sqrt(2.0 / (layer_inputs_ + 1));
    for (int i = 0; i < weights.rows(); ++i) {
        for (int j = 0; j < weights.cols(); ++j) {
            assertTrue(std::abs(weights(i, j)) < 3 * stddev, "Weight out of Xavier range", __FILE__, __LINE__);
        }
        assertTrue(std::abs(biases(i)) < 3 * stddev, "Bias out of Xavier range", __FILE__, __LINE__);
    }
    ++passed_tests_;
    std::cout << "Passed" << std::endl;
}

void NeuralNetworkTest::testLayerForward()
{
    std::cout << "Running testLayerForward... ";
    ++total_tests_;
    Layer layer(layer_inputs_, layer_neurons_, seed_);
    Eigen::VectorXd input(layer_inputs_);
    input.setConstant(1.0);
    auto output = layer.forward(input);

    assertTrue(output.size() == layer_neurons_, "Incorrect output size", __FILE__, __LINE__);

    for (int i = 0; i < output.size(); ++i) {
        assertTrue(output(i) >= 0.0 && output(i) <= 1.0, "Output not in sigmoid range", __FILE__, __LINE__);
    }

    const auto& activations = layer.get_activations();
    for (int i = 0; i < output.size(); ++i) {
        assertApprox(activations(i), output(i), TOL, "Stored activation mismatch", __FILE__, __LINE__);
    }

    ++passed_tests_;
    std::cout << "Passed" << std::endl;
}

void NeuralNetworkTest::testLayerGradients()
{
    std::cout << "Running testLayerGradients... ";
    ++total_tests_;
    Layer layer(layer_inputs_, layer_neurons_, seed_);
    Eigen::VectorXd input(layer_inputs_);

    // Initialize input dynamically based on layer_inputs_
    for (int i = 0; i < layer_inputs_; ++i) {
        input(i) = (i % 2 == 0) ? 1.0 : -1.0; // Alternating 1.0, -1.0
    }
    layer.forward(input);

    Eigen::VectorXd deltas(layer_neurons_);
    deltas.setConstant(0.1); // Simple deltas

    Eigen::MatrixXd weight_grads;
    Eigen::VectorXd bias_grads;
    layer.compute_gradients(deltas, weight_grads, bias_grads);

    assertTrue(weight_grads.rows() == layer_neurons_ && weight_grads.cols() == layer_inputs_,
        "Incorrect weight gradient size", __FILE__, __LINE__);
    assertTrue(bias_grads.size() == layer_neurons_, "Incorrect bias gradient size", __FILE__, __LINE__);

    // Use sigmoid_prime for derivative to be activation-agnostic
    const auto& zs = layer.get_pre_activations();
    Eigen::VectorXd sigmoid_derivs = sigmoid_prime(zs); // From Network.hpp

    for (int i = 0; i < layer_neurons_; ++i) {
        assertApprox(bias_grads(i), deltas(i) * sigmoid_derivs(i), TOL, "Bias gradient incorrect", __FILE__, __LINE__);
        for (int j = 0; j < layer_inputs_; ++j) {
            assertApprox(weight_grads(i, j), deltas(i) * sigmoid_derivs(i) * input(j), TOL,
                "Weight gradient incorrect", __FILE__, __LINE__);
        }
    }

    ++passed_tests_;
    std::cout << "Passed" << std::endl;
}

void NeuralNetworkTest::testLayerUpdateParameters()
{
    std::cout << "Running testLayerUpdateParameters... ";
    ++total_tests_;
    Layer layer(layer_inputs_, layer_neurons_, seed_);

    Eigen::MatrixXd weight_grads(layer_neurons_, layer_inputs_);
    weight_grads.setConstant(0.1);
    Eigen::VectorXd bias_grads(layer_neurons_);
    bias_grads.setConstant(0.1);

    auto old_weights = layer.get_weights();
    auto old_biases = layer.get_biases();

    layer.update_parameters(weight_grads, bias_grads);

    const auto& new_weights = layer.get_weights();
    const auto& new_biases = layer.get_biases();

    for (int i = 0; i < layer_neurons_; ++i) {
        assertApprox(new_biases(i), old_biases(i) - bias_grads(i), TOL, "Bias update incorrect", __FILE__, __LINE__);
        for (int j = 0; j < layer_inputs_; ++j) {
            assertApprox(new_weights(i, j), old_weights(i, j) - weight_grads(i, j), TOL,
                "Weight update incorrect", __FILE__, __LINE__);
        }
    }

    ++passed_tests_;
    std::cout << "Passed" << std::endl;
}

void NeuralNetworkTest::testNetworkConstructor()
{
    std::cout << "Running testNetworkConstructor... ";
    ++total_tests_;
    Network net(network_sizes_);

    assertTrue(net.evaluate({}, 0).first == 0, "Evaluate on empty data should return 0", __FILE__, __LINE__);

    auto [nabla_b, nabla_w] = net.backprop(Eigen::VectorXd(network_sizes_[0]), Eigen::VectorXd(network_sizes_.back()), 0);
    assertTrue(nabla_b.size() == network_sizes_.size() - 1 && nabla_w.size() == network_sizes_.size() - 1,
        "Incorrect number of layers", __FILE__, __LINE__);

    ++passed_tests_;
    std::cout << "Passed" << std::endl;
}

void NeuralNetworkTest::testNetworkFeedforward()
{
    std::cout << "Running testNetworkFeedforward... ";
    ++total_tests_;
    Network net(network_sizes_);
    Eigen::VectorXd input(network_sizes_[0]);
    input.setConstant(1.0);
    auto output = net.feedforward(input);

    assertTrue(output.size() == network_sizes_.back(), "Incorrect output size", __FILE__, __LINE__);

    for (int i = 0; i < output.size(); ++i) {
        assertTrue(output(i) >= 0.0 && output(i) <= 1.0, "Output not in sigmoid range", __FILE__, __LINE__);
    }

    ++passed_tests_;
    std::cout << "Passed" << std::endl;
}

void NeuralNetworkTest::testNetworkBackprop()
{
    std::cout << "Running testNetworkBackprop... ";
    ++total_tests_;
    Network net(network_sizes_);

    Eigen::VectorXd input(network_sizes_[0]);
    input.setConstant(1.0);
    Eigen::VectorXd target(network_sizes_.back());
    target.setZero();
    target(0) = 1.0;

    auto [nabla_b, nabla_w] = net.backprop(input, target, 1);

    assertTrue(nabla_b.size() == network_sizes_.size() - 1 && nabla_w.size() == network_sizes_.size() - 1,
        "Incorrect gradient vector sizes", __FILE__, __LINE__);

    for (size_t i = 0; i < nabla_b.size(); ++i) {
        assertTrue(nabla_b[i].size() == network_sizes_[i + 1], "Incorrect bias gradient size", __FILE__, __LINE__);
        assertTrue(nabla_w[i].rows() == network_sizes_[i + 1] && nabla_w[i].cols() == network_sizes_[i],
            "Incorrect weight gradient size", __FILE__, __LINE__);
    }

    ++passed_tests_;
    std::cout << "Passed" << std::endl;
}

/**
 * @brief Tests Network SGD on a generalized XOR-like dataset.
 */
void NeuralNetworkTest::testNetworkSGD() {
    std::cout << "Running testNetworkSGD... ";
    ++total_tests_;

    // Use fixed network size for XOR-like dataset
    std::vector<int> xor_sizes = { 2, 3, 2 }; // Input: 2, Hidden: 3, Output: 2
    Network net(xor_sizes);

    // Generate XOR-like dataset
    std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> training_data;
    std::vector<std::pair<Eigen::VectorXd, int>> test_data;
    generateXORLikeDataset(training_data, test_data);

    // Run SGD and evaluate
    auto [initial_correct, loss] = net.evaluate(test_data, test_data.size());
    net.SGD(training_data, 400, 2, 1.0, &test_data);
    int final_correct = net.evaluate(test_data, test_data.size()).first;

    assertTrue(final_correct >= initial_correct, "SGD did not improve accuracy", __FILE__, __LINE__);
   
    ++passed_tests_;
    std::cout << "Passed" << std::endl;

    // Note: Future test cases (e.g., AND, OR, MNIST) can be added by extending
    // generateXORLikeDataset or creating new dataset generation functions.
}

bool NeuralNetworkTest::runAllTests()
{
    passed_tests_ = 0;
    total_tests_ = 0;
    testLayerConstructor();
    testLayerForward();
    testLayerGradients();
    testLayerUpdateParameters();
    testNetworkConstructor();
    testNetworkFeedforward();
    testNetworkBackprop();
    testNetworkSGD();
    std::cout << "Test Summary: " << passed_tests_ << "/" << total_tests_ << " tests passed" << std::endl;
    return passed_tests_ == total_tests_;
}

void NeuralNetworkTest::generateXORLikeDataset(std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>& training_data, std::vector<std::pair<Eigen::VectorXd, int>>& test_data)
{
    const int input_size = 2;
    const int output_size = 2;
    const int num_samples = 4; // All XOR input combinations

    training_data.resize(num_samples);
    test_data.resize(num_samples);

    // XOR-like dataset: inputs [0,0], [0,1], [1,0], [1,1]
    // Outputs: [1,0] for even sum (0 or 2), [0,1] for odd sum (1)
    for (int i = 0; i < num_samples; ++i) {
        training_data[i].first = Eigen::VectorXd(input_size);
        test_data[i].first = Eigen::VectorXd(input_size);
        for (int j = 0; j < input_size; ++j) {
            double value = ((i >> j) & 1) ? 1.0 : 0.0;
            training_data[i].first(j) = value;
            test_data[i].first(j) = value;
        }

        training_data[i].second = Eigen::VectorXd(output_size);
        training_data[i].second.setZero();
        int sum = static_cast<int>(training_data[i].first.sum());
        int target_idx = sum % 2; // Even sum -> 0, odd sum -> 1
        training_data[i].second(target_idx) = 1.0;
        test_data[i].second = target_idx;
    }
}

