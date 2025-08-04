#include "NeuralNetworkTest.hpp"
#include "LegacyFuncs.h"
#include"utils.h"
//#include"Network.hpp"
#include <iostream>
#include <cmath>
#include <cassert>
#include <iomanip>
#include <map>

NeuralNetworkTest::NeuralNetworkTest(int layer_inputs, int layer_neurons, unsigned int seed, const std::vector<int>& network_sizes, Network::NeuronType neuron_type) :
    layer_inputs_(layer_inputs), layer_neurons_(layer_neurons), seed_(seed),
    network_sizes_(network_sizes), passed_tests_(0), total_tests_(0), neuron_type_(neuron_type)
{
    // Dynamically create activation based on neuron_type
    switch (neuron_type_) {
    case Network::NeuronType::SIGMOID:
        activation_ = std::make_unique<SigmoidActivation>();
        break;
    default:
        throw std::runtime_error("Unsupported neuron type");
    }

    // Initialize computation contexts
    cpuContext = std::make_unique<CPUComputationContext>();
    gpuContext = std::make_unique<GPUComputationContext>();
    computeContexts = { cpuContext.get(), gpuContext.get() };
}

NeuralNetworkTest::~NeuralNetworkTest()
{
    // Smart pointers automatically clean up cpuContext and gpuContext
}

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

Eigen::VectorXd NeuralNetworkTest::computeActivation(const Eigen::VectorXd& z) const
{
    Eigen::VectorXd result(z.size());
    switch (neuron_type_) {
    case Network::NeuronType::SIGMOID:
        result = activation_->activate(z);
        break;
    default:
        throw std::runtime_error("Unsupported neuron type in computeActivation");
    }
    return result;
}

Eigen::VectorXd NeuralNetworkTest::computeActivationDerivative(const Eigen::VectorXd& z) const
{
    if (z.size() != layer_neurons_) {
        throw std::runtime_error("Invalid pre-activation size in computeActivationDerivative");
    }

    Eigen::VectorXd result(z.size());

    switch (neuron_type_) {
    case Network::NeuronType::SIGMOID:
        result = activation_->derivative(nullptr, &z);
        break;
    default:
        throw std::runtime_error("Unsupported neuron type in computeActivationDerivative");
    }
    return result;
}

bool NeuralNetworkTest::testLayerConstructor() {
    std::cout << "----- Running testLayerConstructor... -----\n";
    ++total_tests_;
    bool all_passed = true;
    std::string contextName;
    std::string errorMsg;
    Eigen::MatrixXd cpu_weights, gpu_weights;
    Eigen::VectorXd cpu_biases, gpu_biases;

    // Test CPU and GPU contexts
    for (auto context : computeContexts) {
        if (dynamic_cast<CPUComputationContext*>(context)) {
            contextName = "CPUContext";
        }
        else if (dynamic_cast<GPUComputationContext*>(context)) {
            contextName = "GPUContext";
        }
        else {
            std::cerr << "Unknown computation context in testLayerConstructor\n";
            throw std::runtime_error("Unknown computation context");
        }

        std::cout << "Test: " << contextName << "\n";

        Layer layer(layer_inputs_, layer_neurons_, activation_.get(), context, seed_);

        // Check weight matrix size
        const auto& weights = layer.get_weights();
        errorMsg = contextName + ": Incorrect weight matrix size";
        assertTrue(weights.rows() == layer_neurons_ && weights.cols() == layer_inputs_,
            errorMsg, __FILE__, __LINE__);

        // Check bias vector size
        const auto& biases = layer.get_biases();
        errorMsg = contextName + ": Incorrect bias vector size";
        assertTrue(biases.size() == layer_neurons_, errorMsg, __FILE__, __LINE__);

        // Check Xavier initialization (values within 3 * stddev and mean near 0)
        //double stddev = std::sqrt(2.0 / (layer_inputs_ + layer_neurons_));
        double stddev = std::sqrt(2.0 / (layer_inputs_ + 1));
        errorMsg = contextName + ": Weight out of Xavier range";
        double weight_mean = weights.mean();
        for (int i = 0; i < weights.rows(); ++i) {
            for (int j = 0; j < weights.cols(); ++j) {
                assertTrue(std::abs(weights(i, j)) < 3 * stddev, errorMsg, __FILE__, __LINE__);
            }
        }
        errorMsg = contextName + ": Weight mean not near zero";
        //assertApprox(weight_mean, 0.0, TOL * 10, errorMsg, __FILE__, __LINE__);
        assertApprox(weight_mean, 0.0, 0.5, errorMsg, __FILE__, __LINE__);

        errorMsg = contextName + ": Bias out of Xavier range";
        double bias_mean = biases.mean();
        for (int i = 0; i < biases.size(); ++i) {
            assertTrue(std::abs(biases(i)) < 3 * stddev, errorMsg, __FILE__, __LINE__);
        }
        errorMsg = contextName + ": Bias mean not near zero";
        //assertApprox(bias_mean, 0.0, TOL * 10, errorMsg, __FILE__, __LINE__);
        assertApprox(bias_mean, 0.0, 0.5, errorMsg, __FILE__, __LINE__);

        // Store weights and biases for CPU-GPU comparison
        if (contextName == "CPUContext") {
            cpu_weights = weights;
            cpu_biases = biases;
        }
        else {
            gpu_weights = weights;
            gpu_biases = biases;
        }

        // Print details
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Weights:\n" << weights << "\n";
        std::cout << "Weight Mean: " << weight_mean << "\n";
        std::cout << "Biases: " << biases.transpose() << "\n";
        std::cout << "Bias Mean: " << bias_mean << "\n";
        std::cout << "Test: " << contextName << " Passed\n\n";
    }

    // Compare CPU and GPU weights and biases (same seed should produce identical results)
    std::cout << "Test: Compare CPU and GPU weights and biases\n";
    bool passed = true;
    errorMsg = "CPU and GPU weights differ";
    for (int i = 0; i < cpu_weights.rows(); ++i) {
        for (int j = 0; j < cpu_weights.cols(); ++j) {
            if (std::abs(cpu_weights(i, j) - gpu_weights(i, j)) > TOL) {
                passed = false;
                assertApprox(cpu_weights(i, j), gpu_weights(i, j), TOL, errorMsg, __FILE__, __LINE__);
            }
        }
    }
    errorMsg = "CPU and GPU biases differ";
    for (int i = 0; i < cpu_biases.size(); ++i) {
        if (std::abs(cpu_biases(i) - gpu_biases(i)) > TOL) {
            passed = false;
            assertApprox(cpu_biases(i), gpu_biases(i), TOL, errorMsg, __FILE__, __LINE__);
        }
    }

    // Print comparison results
    std::cout << "Test: Compare CPU and GPU " << (passed ? "Passed" : "Failed") << "\n\n";

    if (!passed) {
        all_passed = false;
        std::cerr << "Test: Compare CPU and GPU failed\n";
    }

    if (all_passed) {
        ++passed_tests_;
        std::cout << "----- testLayerConstructor Passed -----\n\n";
    }
    else {
        std::cout << "----- testLayerConstructor Failed -----\n\n";
    }
    return all_passed;
}

bool NeuralNetworkTest::testLayerForward() {
    std::cout << "----- Running testLayerForward... -----\n";
    ++total_tests_;
    bool all_passed = true;
    std::string contextName;
    std::string errorMsg;

    // Test 1: CPU and GPU context with random seed
    for (auto context : computeContexts) {
        if (dynamic_cast<CPUComputationContext*>(context)) {
            contextName = "CPUContext";
        }
        else if (dynamic_cast<GPUComputationContext*>(context)) {
            contextName = "GPUContext";
        }
        else {
            std::cerr << "Unknown computation context in testLayerForward\n";
            throw std::runtime_error("Unknown computation context");
        }

        std::cout << "Test: " << contextName << " with random seed\n";
        std::random_device rd;
        unsigned int random_seed = rd();
        std::mt19937 rng(random_seed);
        std::normal_distribution<double> dist(0.0, 1.0);

        Layer layer(layer_inputs_, layer_neurons_, activation_.get(), context, random_seed);

        // Set known weights and biases
        Eigen::MatrixXd weights(layer_neurons_, layer_inputs_);
        for (int i = 0; i < layer_neurons_; ++i) {
            for (int j = 0; j < layer_inputs_; ++j) {
                weights(i, j) = dist(rng);
            }
        }
        Eigen::VectorXd biases(layer_neurons_);
        for (int i = 0; i < layer_neurons_; ++i) {
            biases(i) = dist(rng);
        }
        layer.set_weights(weights);
        layer.set_biases(biases);

        // Random input
        Eigen::VectorXd input(layer_inputs_);
        for (int i = 0; i < layer_inputs_; ++i) {
            input(i) = dist(rng);
        }

        // Compute expected output: z = W * input + b, a = activation(z)
        Eigen::VectorXd z = weights * input + biases;
        Eigen::VectorXd expected = computeActivation(z);

        // Run forward pass
        Eigen::VectorXd output = layer.forward(input);

        // Check output size
        errorMsg = contextName + ": Incorrect output size";
        assertTrue(output.size() == layer_neurons_, errorMsg, __FILE__, __LINE__);

        // Check sigmoid range
        errorMsg = contextName + ": Output not in sigmoid range";
        for (int i = 0; i < output.size(); ++i) {
            assertTrue(output(i) >= 0.0 && output(i) <= 1.0, errorMsg, __FILE__, __LINE__);
        }

        // Check output against expected
        bool passed = true;
        errorMsg = contextName + ": Output mismatch";
        for (int i = 0; i < output.size(); ++i) {
            if (std::abs(output(i) - expected(i)) > TOL) {
                passed = false;
                assertApprox(output(i), expected(i), TOL, errorMsg, __FILE__, __LINE__);
            }
        }

        // Check stored activations
        errorMsg = contextName + ": Stored activation mismatch";
        const auto& activations = layer.get_activations();
        for (int i = 0; i < output.size(); ++i) {
            assertApprox(activations(i), output(i), TOL, errorMsg, __FILE__, __LINE__);
        }

        // Print details
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Input: " << input.transpose() << "\n";
        std::cout << "Weights:\n" << weights << "\n";
        std::cout << "Biases: " << biases.transpose() << "\n";
        std::cout << "Output: " << output.transpose() << "\n";
        std::cout << "Expected: " << expected.transpose() << "\n";
        std::cout << "Test: " << contextName << " " << (passed ? "Passed" : "Failed") << "\n\n";

        if (!passed) {
            all_passed = false;
            std::cerr << "Test: " << contextName << " failed\n";
            return false;
        }
    }

    // Test 2: Compare CPU and GPU contexts with fixed seed and input
    std::cout << "Test: Compare CPU and GPU contexts with fixed seed\n";
    Eigen::VectorXd cpu_output, gpu_output;
    Eigen::MatrixXd weights(layer_neurons_, layer_inputs_);
    Eigen::VectorXd biases(layer_neurons_);
    Eigen::VectorXd input(layer_inputs_);
    unsigned int fixed_seed = 42;

    // Set up weights, biases, and input
    std::mt19937 rng(fixed_seed);
    std::normal_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < layer_neurons_; ++i) {
        for (int j = 0; j < layer_inputs_; ++j) {
            weights(i, j) = dist(rng);
        }
        biases(i) = dist(rng);
    }
    for (int i = 0; i < layer_inputs_; ++i) {
        input(i) = dist(rng);
    }

    // Compute expected output
    Eigen::VectorXd z = weights * input + biases;
    Eigen::VectorXd expected = computeActivation(z);

    // Run forward pass for both contexts
    bool passed = true;
    for (auto context : computeContexts) {
        if (dynamic_cast<CPUComputationContext*>(context)) {
            contextName = "CPUContext";
        }
        else if (dynamic_cast<GPUComputationContext*>(context)) {
            contextName = "GPUContext";
        }
        else {
            std::cerr << "Unknown computation context in testLayerForward\n";
            throw std::runtime_error("Unknown computation context");
        }
        Layer layer(layer_inputs_, layer_neurons_, activation_.get(), context, fixed_seed);
        layer.set_weights(weights);
        layer.set_biases(biases);

        Eigen::VectorXd output = layer.forward(input);
        if (contextName == "CPUContext") {
            cpu_output = output;
        }
        else {
            gpu_output = output;
        }

        // Check output size
        errorMsg = contextName + ": Incorrect output size";
        assertTrue(output.size() == layer_neurons_, errorMsg, __FILE__, __LINE__);

        // Check sigmoid range
        errorMsg = contextName + ": Output not in sigmoid range";
        for (int i = 0; i < output.size(); ++i) {
            assertTrue(output(i) >= 0.0 && output(i) <= 1.0, errorMsg, __FILE__, __LINE__);
        }

        // Check output against expected
        errorMsg = contextName + ": Output mismatch";
        for (int i = 0; i < output.size(); ++i) {
            if (std::abs(output(i) - expected(i)) > TOL) {
                passed = false;
                assertApprox(output(i), expected(i), TOL, errorMsg, __FILE__, __LINE__);
            }
        }

        // Check stored activations
        errorMsg = contextName + ": Stored activation mismatch";
        const auto& activations = layer.get_activations();
        for (int i = 0; i < output.size(); ++i) {
            assertApprox(activations(i), output(i), TOL, errorMsg, __FILE__, __LINE__);
        }
    }

    // Compare CPU and GPU outputs
    errorMsg = "CPU and GPU outputs differ";
    for (int i = 0; i < cpu_output.size(); ++i) {
        if (std::abs(cpu_output(i) - gpu_output(i)) > TOL) {
            passed = false;
            assertApprox(cpu_output(i), gpu_output(i), TOL, errorMsg, __FILE__, __LINE__);
        }
    }

    // Print details
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Input: " << input.transpose() << "\n";
    std::cout << "Weights:\n" << weights << "\n";
    std::cout << "Biases: " << biases.transpose() << "\n";
    std::cout << "CPU Output: " << cpu_output.transpose() << "\n";
    std::cout << "GPU Output: " << gpu_output.transpose() << "\n";
    std::cout << "Expected: " << expected.transpose() << "\n";
    std::cout << "Test: Compare CPU and GPU " << (passed ? "Passed" : "Failed") << "\n\n";

    if (!passed) {
        all_passed = false;
        std::cerr << "Test: Compare CPU and GPU failed\n";
    }

    if (all_passed) {
        ++passed_tests_;
        std::cout << "----- testLayerForward Passed -----\n\n";
    }

    else {
        std::cout << "----- testLayerForward Failed -----\n\n";
    }

    return all_passed;
}

bool NeuralNetworkTest::testLayerGradients()
{
    std::cout << "----- Running testLayerGradients... -----\n";
    ++total_tests_;
    bool all_passed = true;
    std::string contextName;
    std::string errorMsg;
    Eigen::MatrixXd cpu_weight_grads, gpu_weight_grads;
    Eigen::VectorXd cpu_bias_grads, gpu_bias_grads;

    // Test CPU and GPU contexts
    for (auto context : computeContexts) {
        if (dynamic_cast<CPUComputationContext*>(context)) {
            contextName = "CPUContext";
        }
        else if (dynamic_cast<GPUComputationContext*>(context)) {
            contextName = "GPUContext";
        }
        else {
            std::cerr << "Unknown computation context in testLayerGradients\n";
            throw std::runtime_error("Unknown computation context");
        }

        std::cout << "Test: " << contextName << "\n";

        // Initialize layer with random weights and biases
        Layer layer(layer_inputs_, layer_neurons_, activation_.get(), context, seed_);
        std::mt19937 rng(seed_);
        std::normal_distribution<double> dist(0.0, std::sqrt(2.0 / (layer_inputs_ + 1)));
        Eigen::MatrixXd weights(layer_neurons_, layer_inputs_);
        for (int i = 0; i < layer_neurons_; ++i) {
            for (int j = 0; j < layer_inputs_; ++j) {
                weights(i, j) = dist(rng);
            }
        }
        Eigen::VectorXd biases(layer_neurons_);
        for (int i = 0; i < layer_neurons_; ++i) {
            biases(i) = dist(rng);
        }
        layer.set_weights(weights);
        layer.set_biases(biases);

        // Initialize input
        Eigen::VectorXd input(layer_inputs_);
        for (int i = 0; i < layer_inputs_; ++i) {
            input(i) = (i % 2 == 0) ? 1.0 : -1.0; // Alternating 1.0, -1.0
        }

        // Run forward pass
        Eigen::VectorXd output = layer.forward(input);

        // Check pre-activations
        const auto& zs = layer.get_pre_activations();
        Eigen::VectorXd expected_z = weights * input + biases;
        errorMsg = contextName + ": Incorrect pre-activations";
        for (int i = 0; i < zs.size(); ++i) {
            assertApprox(zs(i), expected_z(i), TOL, errorMsg, __FILE__, __LINE__);
        }

        // Initialize deltas
        Eigen::VectorXd deltas(layer_neurons_);
        deltas.setConstant(0.1); // Simple deltas

        // Compute gradients
        Eigen::MatrixXd weight_grads;
        Eigen::VectorXd bias_grads;
        layer.compute_gradients(deltas, weight_grads, bias_grads);

        // Check gradient sizes
        errorMsg = contextName + ": Incorrect weight gradient size";
        assertTrue(weight_grads.rows() == layer_neurons_ && weight_grads.cols() == layer_inputs_,
            errorMsg, __FILE__, __LINE__);
        errorMsg = contextName + ": Incorrect bias gradient size";
        assertTrue(bias_grads.size() == layer_neurons_, errorMsg, __FILE__, __LINE__);

        // Compute expected gradients using activation derivative
        Eigen::VectorXd derivs = computeActivationDerivative(zs);
        Eigen::MatrixXd expected_weight_grads(layer_neurons_, layer_inputs_);
        Eigen::VectorXd expected_bias_grads(layer_neurons_);
        for (int i = 0; i < layer_neurons_; ++i) {
            expected_bias_grads(i) = deltas(i) * derivs(i);
            for (int j = 0; j < layer_inputs_; ++j) {
                expected_weight_grads(i, j) = deltas(i) * derivs(i) * input(j);
            }
        }

        // Check gradients
        bool passed = true;
        errorMsg = contextName + ": Weight gradient incorrect";
        for (int i = 0; i < layer_neurons_; ++i) {
            for (int j = 0; j < layer_inputs_; ++j) {
                if (std::abs(weight_grads(i, j) - expected_weight_grads(i, j)) > TOL) {
                    passed = false;
                    assertApprox(weight_grads(i, j), expected_weight_grads(i, j), TOL, errorMsg, __FILE__, __LINE__);
                }
            }
        }

        errorMsg = contextName + ": Bias gradient incorrect";
        for (int i = 0; i < layer_neurons_; ++i) {
            if (std::abs(bias_grads(i) - expected_bias_grads(i)) > TOL) {
                passed = false;
                assertApprox(bias_grads(i), expected_bias_grads(i), TOL, errorMsg, __FILE__, __LINE__);
            }
        }

        // Store gradients for CPU-GPU comparison
        if (contextName == "CPUContext") {
            cpu_weight_grads = weight_grads;
            cpu_bias_grads = bias_grads;
        }
        else {
            gpu_weight_grads = weight_grads;
            gpu_bias_grads = bias_grads;
        }

        // Print details
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Input: " << input.transpose() << "\n";
        std::cout << "Weights:\n" << weights << "\n";
        std::cout << "Biases: " << biases.transpose() << "\n";
        std::cout << "Deltas: " << deltas.transpose() << "\n";
        std::cout << "Pre-activations: " << zs.transpose() << "\n";
        //std::cout << "Derivatives: " << derivs.transpose() << "\n";
        std::cout << "Weight Gradients:\n" << weight_grads << "\n";
        std::cout << "Bias Gradients: " << bias_grads.transpose() << "\n";
        std::cout << "Expected Weight Gradients:\n" << expected_weight_grads << "\n";
        std::cout << "Expected Bias Gradients: " << expected_bias_grads.transpose() << "\n";
        std::cout << "Test: " << contextName << " " << (passed ? "Passed" : "Failed") << "\n\n";

        if (!passed) {
            all_passed = false;
            std::cerr << "Test: " << contextName << " failed\n";
            return false;
        }
    }

    // Compare CPU and GPU gradients
    std::cout << "Test: Compare CPU and GPU gradients\n";
    bool passed = true;
    errorMsg = "CPU and GPU weight gradients differ";
    for (int i = 0; i < cpu_weight_grads.rows(); ++i) {
        for (int j = 0; j < cpu_weight_grads.cols(); ++j) {
            if (std::abs(cpu_weight_grads(i, j) - gpu_weight_grads(i, j)) > TOL) {
                passed = false;
                assertApprox(cpu_weight_grads(i, j), gpu_weight_grads(i, j), TOL, errorMsg, __FILE__, __LINE__);
            }
        }
    }
    errorMsg = "CPU and GPU bias gradients differ";
    for (int i = 0; i < cpu_bias_grads.size(); ++i) {
        if (std::abs(cpu_bias_grads(i) - gpu_bias_grads(i)) > TOL) {
            passed = false;
            assertApprox(cpu_bias_grads(i), gpu_bias_grads(i), TOL, errorMsg, __FILE__, __LINE__);
        }
    }

    // Print comparison results
    std::cout << "Test: Compare CPU and GPU " << (passed ? "Passed" : "Failed") << "\n\n";

    if (!passed) {
        all_passed = false;
        std::cerr << "Test: Compare CPU and GPU failed\n";
    }

    if (all_passed) {
        ++passed_tests_;
        std::cout << "----- testLayerGradients Passed -----\n\n";
    }
    else {
        std::cout << "----- testLayerGradients Failed -----\n\n";
    }
    return all_passed;
}

bool NeuralNetworkTest::testLayerUpdateParameters() {
    std::cout << "----- Running testLayerUpdateParameters... -----\n";
    ++total_tests_;
    bool all_passed = true;
    std::string contextName;
    std::string errorMsg;
    std::map<std::string, Eigen::MatrixXd> weights_map, updated_weights_map;
    std::map<std::string, Eigen::VectorXd> biases_map, updated_biases_map;

    // Test CPU and GPU contexts
    for (auto context : computeContexts) {
        if (dynamic_cast<CPUComputationContext*>(context)) {
            contextName = "CPUContext";
        }
        else if (dynamic_cast<GPUComputationContext*>(context)) {
            contextName = "GPUContext";
        }
        else {
            std::cerr << "Unknown computation context in testLayerUpdateParameters\n";
            throw std::runtime_error("Unknown computation context");
        }

        std::cout << "Test: " << contextName << "\n";

        // Initialize layer with random weights and biases
        Layer layer(layer_inputs_, layer_neurons_, activation_.get(), context, seed_);
        std::mt19937 rng(seed_);
        std::normal_distribution<double> dist(0.0, std::sqrt(2.0 / (layer_inputs_ + 1)));
        Eigen::MatrixXd weights(layer_neurons_, layer_inputs_);
        for (int i = 0; i < layer_neurons_; ++i) {
            for (int j = 0; j < layer_inputs_; ++j) {
                weights(i, j) = dist(rng);
            }
        }
        Eigen::VectorXd biases(layer_neurons_);
        for (int i = 0; i < layer_neurons_; ++i) {
            biases(i) = dist(rng);
        }
        layer.set_weights(weights);
        layer.set_biases(biases);

        // Initialize gradients
        Eigen::MatrixXd weight_grads(layer_neurons_, layer_inputs_);
        weight_grads.setConstant(0.1);
        Eigen::VectorXd bias_grads(layer_neurons_);
        bias_grads.setConstant(0.1);

        // Store original parameters
        auto old_weights = layer.get_weights();
        auto old_biases = layer.get_biases();

        // Update parameters
        layer.update_parameters(weight_grads, bias_grads);

        // Check updated parameters
        const auto& new_weights = layer.get_weights();
        const auto& new_biases = layer.get_biases();
        bool passed = true;
        errorMsg = contextName + ": Weight update incorrect";
        for (int i = 0; i < layer_neurons_; ++i) {
            for (int j = 0; j < layer_inputs_; ++j) {
                if (std::abs(new_weights(i, j) - (old_weights(i, j) - weight_grads(i, j))) > TOL) {
                    passed = false;
                    assertApprox(new_weights(i, j), old_weights(i, j) - weight_grads(i, j), TOL, errorMsg, __FILE__, __LINE__);
                }
            }
        }
        errorMsg = contextName + ": Bias update incorrect";
        for (int i = 0; i < layer_neurons_; ++i) {
            if (std::abs(new_biases(i) - (old_biases(i) - bias_grads(i))) > TOL) {
                passed = false;
                assertApprox(new_biases(i), old_biases(i) - bias_grads(i), TOL, errorMsg, __FILE__, __LINE__);
            }
        }

        // Store weights and biases for comparison
        weights_map[contextName] = old_weights;
        updated_weights_map[contextName] = new_weights;
        biases_map[contextName] = old_biases;
        updated_biases_map[contextName] = new_biases;

        // Print details
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Original Weights:\n" << old_weights << "\n";
        std::cout << "Weight Gradients:\n" << weight_grads << "\n";
        std::cout << "Updated Weights:\n" << new_weights << "\n";
        std::cout << "Original Biases: " << old_biases.transpose() << "\n";
        std::cout << "Bias Gradients: " << bias_grads.transpose() << "\n";
        std::cout << "Updated Biases: " << new_biases.transpose() << "\n";
        std::cout << "Test: " << contextName << " " << (passed ? "Passed" : "Failed") << "\n\n";

        if (!passed) {
            all_passed = false;
            std::cerr << "Test: " << contextName << " failed\n";
            return false;
        }
    }

    // Compare CPU and GPU parameters
    std::cout << "Test: Compare CPU and GPU parameters\n";
    bool passed = true;
    errorMsg = "CPU and GPU weights differ before update";
    for (int i = 0; i < layer_neurons_; ++i) {
        for (int j = 0; j < layer_inputs_; ++j) {
            if (std::abs(weights_map["CPUContext"](i, j) - weights_map["GPUContext"](i, j)) > TOL) {
                passed = false;
                assertApprox(weights_map["CPUContext"](i, j), weights_map["GPUContext"](i, j), TOL, errorMsg, __FILE__, __LINE__);
            }
        }
    }

    errorMsg = "CPU and GPU biases differ before update";
    for (int i = 0; i < layer_neurons_; ++i) {
        if (std::abs(biases_map["CPUContext"](i) - biases_map["GPUContext"](i)) > TOL) {
            passed = false;
            assertApprox(biases_map["CPUContext"](i), biases_map["GPUContext"](i), TOL, errorMsg, __FILE__, __LINE__);
        }
    }

    errorMsg = "CPU and GPU updated weights differ";
    for (int i = 0; i < layer_neurons_; ++i) {
        for (int j = 0; j < layer_inputs_; ++j) {
            if (std::abs(updated_weights_map["CPUContext"](i, j) - updated_weights_map["GPUContext"](i, j)) > TOL) {
                passed = false;
                assertApprox(updated_weights_map["CPUContext"](i, j), updated_weights_map["GPUContext"](i, j), TOL, errorMsg, __FILE__, __LINE__);
            }
        }
    }
    errorMsg = "CPU and GPU updated biases differ";
    for (int i = 0; i < layer_neurons_; ++i) {
        if (std::abs(updated_biases_map["CPUContext"](i) - updated_biases_map["GPUContext"](i)) > TOL) {
            passed = false;
            assertApprox(updated_biases_map["CPUContext"](i), updated_biases_map["GPUContext"](i), TOL, errorMsg, __FILE__, __LINE__);
        }
    }

    // Print comparison results
    std::cout << "Test: Compare CPU and GPU " << (passed ? "Passed" : "Failed") << "\n\n";

    if (!passed) {
        all_passed = false;
        std::cerr << "Test: Compare CPU and GPU failed\n";
    }

    if (all_passed) {
        ++passed_tests_;
        std::cout << "----- testLayerUpdateParameters Passed -----\n";
    }
    else {
        std::cout << "----- testLayerUpdateParameters Failed -----\n";
    }
    return all_passed;
}

bool NeuralNetworkTest::testLayerComputeActivationDerivative()
{
    std::cout << "----- Running testLayerComputeActivationDerivative... -----\n";
    ++total_tests_;
    bool all_passed = true;
    std::string contextName;
    std::string errorMsg;
    std::map<std::string, Eigen::VectorXd> derivatives_map;

    // Test CPU and GPU contexts
    for (auto context : computeContexts) {
        if (dynamic_cast<CPUComputationContext*>(context)) {
            contextName = "CPUContext";
        }
        else if (dynamic_cast<GPUComputationContext*>(context)) {
            contextName = "GPUContext";
        }
        else {
            std::cerr << "Unknown computation context in testLayerComputeActivationDerivative\n";
            throw std::runtime_error("Unknown computation context");
        }

        std::cout << "Test: " << contextName << "\n";

        // Initialize layer
        Layer layer(layer_inputs_, layer_neurons_, activation_.get(), context, seed_);
        
        // Initialize pre-activations and activations
        Eigen::VectorXd pre_activations(layer_neurons_);
        pre_activations << 0.0, 1.0, -1.0; // Simple test values
        Eigen::VectorXd activations = computeActivation(pre_activations);

        // Compute expected derivatives
        Eigen::VectorXd expected_derivatives = computeActivationDerivative(pre_activations);

        // Compute derivatives using context
        Eigen::VectorXd derivatives = context->computeActivationDerivative(activations, pre_activations, activation_.get());
        
        // Check derivative size
        errorMsg = contextName + ": Incorrect derivative size";
        assertTrue(derivatives.size() == layer_neurons_, errorMsg, __FILE__, __LINE__);

        // Check derivative size
        errorMsg = contextName + ": Incorrect derivative size";
        assertTrue(derivatives.size() == layer_neurons_, errorMsg, __FILE__, __LINE__);

        // Check derivatives against expected
        bool passed = true;
        errorMsg = contextName + ": Derivative incorrect";
        for (int i = 0; i < derivatives.size(); ++i) {
            if (std::abs(derivatives(i) - expected_derivatives(i)) > TOL) {
                passed = false;
                assertApprox(derivatives(i), expected_derivatives(i), TOL, errorMsg, __FILE__, __LINE__);
            }
        }

        // Store derivatives for CPU-GPU comparison
        derivatives_map[contextName] = derivatives;

        // Print details
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Pre-activations: " << pre_activations.transpose() << "\n";
        std::cout << "Activations: " << activations.transpose() << "\n";
        std::cout << "Derivatives: " << derivatives.transpose() << "\n";
        std::cout << "Expected Derivatives: " << expected_derivatives.transpose() << "\n";
        std::cout << "Test: " << contextName << " " << (passed ? "Passed" : "Failed") << "\n\n";

        if (!passed) {
            all_passed = false;
            std::cerr << "Test: " << contextName << " failed\n";
            return false;
        }
    }

    // Compare CPU and GPU derivatives
    std::cout << "Test: Compare CPU and GPU derivatives\n";
    bool passed = true;
    errorMsg = "CPU and GPU derivatives differ";
    for (int i = 0; i < derivatives_map["CPUContext"].size(); ++i) {
        if (std::abs(derivatives_map["CPUContext"](i) - derivatives_map["GPUContext"](i)) > TOL) {
            passed = false;
            assertApprox(derivatives_map["CPUContext"](i), derivatives_map["GPUContext"](i), TOL, errorMsg, __FILE__, __LINE__);
        }
    }

    // Print comparison results
    std::cout << "Test: Compare CPU and GPU " << (passed ? "Passed" : "Failed") << "\n\n";

    if (!passed) {
        all_passed = false;
        std::cerr << "Test: Compare CPU and GPU failed\n";
    }

    if (all_passed) {
        ++passed_tests_;
        std::cout << "----- testLayerComputeActivationDerivative Passed -----\n\n";
    }
    else {
        std::cout << "----- testLayerComputeActivationDerivative Failed -----\n\n";
    }
    return all_passed;
}

bool NeuralNetworkTest::testLayerComputeActivationDerivativeGPU()
{
    std::cout << "----- Running testLayerComputeActivationDerivative... -----\n";
    ++total_tests_;
    bool all_passed = true;
    std::string contextName;
    std::string errorMsg;
    std::map<std::string, Eigen::VectorXd> derivatives_map;

    // Test CPU and GPU contexts
    for (auto context : computeContexts) {
        if (dynamic_cast<CPUComputationContext*>(context)) {
            contextName = "CPUContext";
        }
        else if (dynamic_cast<GPUComputationContext*>(context)) {
            contextName = "GPUContext";
        }
        else {
            std::cerr << "Unknown computation context in testLayerComputeActivationDerivative\n";
            throw std::runtime_error("Unknown computation context");
        }

        std::cout << "Test: " << contextName << "\n";

        // Initialize layer
        Layer layer(layer_inputs_, layer_neurons_, activation_.get(), context, seed_);

        // Initialize pre-activations and activations
        Eigen::VectorXd pre_activations(layer_neurons_);
        pre_activations << 0.0, 1.0, -1.0; // Simple test values
        
        Eigen::VectorXd activations = computeActivation(pre_activations);
        layer.set_pre_activations(pre_activations);
        layer.set_activations(activations);

        // Compute expected derivatives
        Eigen::VectorXd expected_derivatives = computeActivationDerivative(pre_activations);

        // Compute derivatives using context
        Eigen::VectorXd derivatives;
        if (contextName == "GPUContext") {
            derivatives = context->computeActivationDerivativeGPU(layer.get_d_activations_(), layer.get_d_pre_activations_(), layer.get_d_dy(), layer.get_d_derivatives(), layer.get_num_neurons(), activation_.get());
        }
        else {
            derivatives = context->computeActivationDerivative(activations, pre_activations, activation_.get());
        }
        
        // Check derivative size
        errorMsg = contextName + ": Incorrect derivative size";
        assertTrue(derivatives.size() == layer_neurons_, errorMsg, __FILE__, __LINE__);

       
        // Check derivatives against expected
        bool passed = true;
        errorMsg = contextName + ": Derivative incorrect";
        for (int i = 0; i < derivatives.size(); ++i) {
            if (std::abs(derivatives(i) - expected_derivatives(i)) > TOL) {
                passed = false;
                assertApprox(derivatives(i), expected_derivatives(i), TOL, errorMsg, __FILE__, __LINE__);
            }
        }

        // Store derivatives for CPU-GPU comparison
        derivatives_map[contextName] = derivatives;

        // Print details
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Pre-activations: " << pre_activations.transpose() << "\n";
        std::cout << "Activations: " << activations.transpose() << "\n";
        std::cout << "Derivatives: " << derivatives.transpose() << "\n";
        std::cout << "Expected Derivatives: " << expected_derivatives.transpose() << "\n";
        std::cout << "Test: " << contextName << " " << (passed ? "Passed" : "Failed") << "\n\n";

        if (!passed) {
            all_passed = false;
            std::cerr << "Test: " << contextName << " failed\n";
            return false;
        }
    }

    // Compare CPU and GPU derivatives
    std::cout << "Test: Compare CPU and GPU derivatives\n";
    bool passed = true;
    errorMsg = "CPU and GPU derivatives differ";
    for (int i = 0; i < derivatives_map["CPUContext"].size(); ++i) {
        if (std::abs(derivatives_map["CPUContext"](i) - derivatives_map["GPUContext"](i)) > TOL) {
            passed = false;
            assertApprox(derivatives_map["CPUContext"](i), derivatives_map["GPUContext"](i), TOL, errorMsg, __FILE__, __LINE__);
        }
    }

    // Print comparison results
    std::cout << "Test: Compare CPU and GPU " << (passed ? "Passed" : "Failed") << "\n\n";

    if (!passed) {
        all_passed = false;
        std::cerr << "Test: Compare CPU and GPU failed\n";
    }

    if (all_passed) {
        ++passed_tests_;
        std::cout << "----- testLayerComputeActivationDerivative Passed -----\n\n";
    }
    else {
        std::cout << "----- testLayerComputeActivationDerivative Failed -----\n\n";
    }
    return all_passed;
}

bool NeuralNetworkTest::testUpdateMiniBatchSimplified()
{
    std::cout << "----- Running testUpdateMiniBatchSimplified... -----\n";
    ++total_tests_;
    bool all_passed = true;
    std::string contextName;
    std::string errorMsg;

    std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> mini_batch = {
        {Eigen::VectorXd::Zero(2), Eigen::VectorXd::Zero(2)},
        {Eigen::VectorXd::Unit(2, 0), Eigen::VectorXd::Unit(2, 1)}
    };
    double eta = 0.1;
    size_t n = 2;
    unsigned int seed = 42;

    std::vector<int> sizes = { 2, 3, 2 };
    std::map<std::string, std::vector<Eigen::MatrixXd>> weights_map;
    std::map<std::string, std::vector<Eigen::VectorXd>> biases_map;

    for (auto context : computeContexts) {
        if (dynamic_cast<CPUComputationContext*>(context)) {
            contextName = "CPUContext";
        }
        else if (dynamic_cast<GPUComputationContext*>(context)) {
            contextName = "GPUContext";
        }
        else {
            std::cerr << "Unknown computation context in testUpdateMiniBatchSimplified\n";
            throw std::runtime_error("Unknown computation context");
        }

        std::cout << "Test: " << contextName << "\n";
        Network net(sizes, 0, Network::LossType::MSE, Network::NeuronType::SIGMOID, context, seed);

        double norm = net.update_mini_batch(mini_batch, eta, n);

        std::vector<Eigen::MatrixXd> weights;
        std::vector<Eigen::VectorXd> biases;
        for (const auto& layer : net.get_layers()) {
            weights.push_back(layer->get_weights());
            biases.push_back(layer->get_biases());
        }

        weights_map[contextName] = weights;
        biases_map[contextName] = biases;

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Gradient Norm: " << norm << "\n";
        for (size_t i = 0; i < net.get_layers().size(); ++i) {
            std::cout << "Layer " << i << " Updated Weights:\n" << weights[i] << "\n";
            std::cout << "Layer " << i << " Updated Biases: " << biases[i].transpose() << "\n";
        }
        std::cout << "Test: " << contextName << " Passed\n";
    }

    std::cout << "\nTest: Manual calculations\n";
    std::vector<Eigen::MatrixXd> manual_weights;
    std::vector<Eigen::VectorXd> manual_biases;
    {
        Network net(sizes, 0.0, Network::LossType::MSE, Network::NeuronType::SIGMOID, computeContexts[0], seed);
        double scale = eta / mini_batch.size();

        std::vector<Eigen::MatrixXd> weights;
        std::vector<Eigen::VectorXd> biases;
        for (const auto& layer : net.get_layers()) {
            weights.push_back(layer->get_weights());
            biases.push_back(layer->get_biases());
        }

        std::vector<Eigen::MatrixXd> weight_grads;
        std::vector<Eigen::VectorXd> bias_grads;
        for (size_t i = 0; i < biases.size(); ++i) {
            weight_grads.emplace_back(Eigen::MatrixXd::Zero(weights[i].rows(), weights[i].cols()));
            bias_grads.emplace_back(Eigen::VectorXd::Zero(biases[i].size()));
        }

        for (const auto& [x, y] : mini_batch) {
            auto [delta_nabla_b, delta_nabla_w] = net.backprop(x, y, n);
            for (size_t i = 0; i < delta_nabla_w.size(); ++i) {
                weight_grads[i] += delta_nabla_w[i];
                bias_grads[i] += delta_nabla_b[i];
            }
        }

        double norm = 0.0;
        for (size_t i = 0; i < weight_grads.size(); ++i) {
            norm += weight_grads[i].squaredNorm();
            norm += bias_grads[i].squaredNorm();
        }
        norm = std::sqrt(norm / mini_batch.size());

        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i] -= scale * weight_grads[i];
            biases[i] -= scale * bias_grads[i];
        }

        manual_weights = weights;
        manual_biases = biases;

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Gradient Norm: " << norm << "\n";
        for (size_t i = 0; i < weights.size(); ++i) {
            std::cout << "Layer " << i << " Updated Weights:\n" << weights[i] << "\n";
            std::cout << "Layer " << i << " Updated Biases: " << biases[i].transpose() << "\n";
        }
        std::cout << "Test: Manual calculations Passed\n";
    }

    std::cout << "\nTest: Compare CPU, GPU, and Manual updated parameters\n";
    bool passed = true;
    for (size_t i = 0; i < sizes.size() - 1; ++i) {
        errorMsg = "CPU and GPU weights differ in layer " + std::to_string(i);
        for (int r = 0; r < weights_map["CPUContext"][i].rows(); ++r) {
            for (int c = 0; c < weights_map["CPUContext"][i].cols(); ++c) {
                double diff = std::abs(weights_map["CPUContext"][i](r, c) - weights_map["GPUContext"][i](r, c));
                if (diff > TOL) {
                    std::cerr << "CPU-GPU weight mismatch in layer " << i << " at (" << r << "," << c << "): "
                        << weights_map["CPUContext"][i](r, c) << " vs "
                        << weights_map["GPUContext"][i](r, c) << " (diff = " << diff << ")\n";
                    passed = false;
                    assertApprox(weights_map["CPUContext"][i](r, c), weights_map["GPUContext"][i](r, c), TOL, errorMsg, __FILE__, __LINE__);
                }
            }
        }
        errorMsg = "CPU and GPU biases differ in layer " + std::to_string(i);
        for (int j = 0; j < biases_map["CPUContext"][i].size(); ++j) {
            double diff = std::abs(biases_map["CPUContext"][i](j) - biases_map["GPUContext"][i](j));
            if (diff > TOL) {
                std::cerr << "CPU-GPU bias mismatch in layer " << i << " at index " << j << ": "
                    << biases_map["CPUContext"][i](j) << " vs "
                    << biases_map["GPUContext"][i](j) << " (diff = " << diff << ")\n";
                passed = false;
                assertApprox(biases_map["CPUContext"][i](j), biases_map["GPUContext"][i](j), TOL, errorMsg, __FILE__, __LINE__);
            }
        }

        errorMsg = "CPU and Manual weights differ in layer " + std::to_string(i);
        for (int r = 0; r < weights_map["CPUContext"][i].rows(); ++r) {
            for (int c = 0; c < weights_map["CPUContext"][i].cols(); ++c) {
                double diff = std::abs(weights_map["CPUContext"][i](r, c) - manual_weights[i](r, c));
                if (diff > TOL) {
                    std::cerr << "CPU-Manual weight mismatch in layer " << i << " at (" << r << "," << c << "): "
                        << weights_map["CPUContext"][i](r, c) << " vs "
                        << manual_weights[i](r, c) << " (diff = " << diff << ")\n";
                    passed = false;
                    assertApprox(weights_map["CPUContext"][i](r, c), manual_weights[i](r, c), TOL, errorMsg, __FILE__, __LINE__);
                }
            }
        }
        errorMsg = "CPU and Manual biases differ in layer " + std::to_string(i);
        for (int j = 0; j < biases_map["CPUContext"][i].size(); ++j) {
            double diff = std::abs(biases_map["CPUContext"][i](j) - manual_biases[i](j));
            if (diff > TOL) {
                std::cerr << "CPU-Manual bias mismatch in layer " << i << " at index " << j << ": "
                    << biases_map["CPUContext"][i](j) << " vs "
                    << manual_biases[i](j) << " (diff = " << diff << ")\n";
                passed = false;
                assertApprox(biases_map["CPUContext"][i](j), manual_biases[i](j), TOL, errorMsg, __FILE__, __LINE__);
            }
        }

        errorMsg = "GPU and Manual weights differ in layer " + std::to_string(i);
        for (int r = 0; r < weights_map["GPUContext"][i].rows(); ++r) {
            for (int c = 0; c < weights_map["GPUContext"][i].cols(); ++c) {
                double diff = std::abs(weights_map["GPUContext"][i](r, c) - manual_weights[i](r, c));
                if (diff > TOL) {
                    std::cerr << "GPU-Manual weight mismatch in layer " << i << " at (" << r << "," << c << "): "
                        << weights_map["GPUContext"][i](r, c) << " vs "
                        << manual_weights[i](r, c) << " (diff = " << diff << ")\n";
                    passed = false;
                    assertApprox(weights_map["GPUContext"][i](r, c), manual_weights[i](r, c), TOL, errorMsg, __FILE__, __LINE__);
                }
            }
        }
        errorMsg = "GPU and Manual biases differ in layer " + std::to_string(i);
        for (int j = 0; j < biases_map["GPUContext"][i].size(); ++j) {
            double diff = std::abs(biases_map["GPUContext"][i](j) - manual_biases[i](j));
            if (diff > TOL) {
                std::cerr << "GPU-Manual bias mismatch in layer " << i << " at index " << j << ": "
                    << biases_map["GPUContext"][i](j) << " vs "
                    << manual_biases[i](j) << " (diff = " << diff << ")\n";
                passed = false;
                assertApprox(biases_map["GPUContext"][i](j), manual_biases[i](j), TOL, errorMsg, __FILE__, __LINE__);
            }
        }
    }
    std::cout << "Test: Compare CPU, GPU, and Manual " << (passed ? "Passed" : "Failed") << "\n\n";

    if (!passed) all_passed = false;

    if (all_passed) {
        ++passed_tests_;
        std::cout << "----- testUpdateMiniBatchSimplified Passed -----\n\n";
    }
    else {
        std::cout << "----- testUpdateMiniBatchSimplified Failed -----\n\n";
    }

    return all_passed;
}

bool NeuralNetworkTest::testBackpropGradientComputation()
{
    std::cout << "----- Running testBackpropGradientComputation... -----\n";
    ++total_tests_;
    bool all_passed = true;
    std::string contextName;
    std::string errorMsg;
    std::map<std::string, std::vector<Eigen::MatrixXd>> nabla_w_map;
    std::map<std::string, std::vector<Eigen::VectorXd>> nabla_b_map;

    std::vector<int> sizes = { 2, 3, 2 };
    unsigned int seed = 42;
    Eigen::VectorXd x(2);
    x << 0.5, 0.3;
    Eigen::VectorXd y(2);
    y << 1.0, 0.0;
    size_t n = 2;

    for (auto context : computeContexts) {
        if (dynamic_cast<CPUComputationContext*>(context)) {
            contextName = "CPUContext";
        }
        else if (dynamic_cast<GPUComputationContext*>(context)) {
            contextName = "GPUContext";
        }
        else {
            std::cerr << "Unknown computation context in testBackpropGradientComputation\n";
            throw std::runtime_error("Unknown computation context");
        }

        std::cout << "Test: " << contextName << "\n";

        Network net(sizes, 0, Network::LossType::MSE, Network::NeuronType::SIGMOID, context, seed);
        auto [nabla_b, nabla_w] = net.backprop(x, y, n);

        errorMsg = contextName + ": Incorrect number of gradient vectors";
        assertTrue(nabla_b.size() == sizes.size() - 1 && nabla_w.size() == sizes.size() - 1,
            errorMsg, __FILE__, __LINE__);

        for (size_t i = 0; i < nabla_b.size(); ++i) {
            errorMsg = contextName + ": Incorrect bias gradient size in layer " + std::to_string(i);
            assertTrue(nabla_b[i].size() == sizes[i + 1], errorMsg, __FILE__, __LINE__);
            errorMsg = contextName + ": Incorrect weight gradient size in layer " + std::to_string(i);
            assertTrue(nabla_w[i].rows() == sizes[i + 1] && nabla_w[i].cols() == sizes[i],
                errorMsg, __FILE__, __LINE__);
        }

        nabla_b_map[contextName] = nabla_b;
        nabla_w_map[contextName] = nabla_w;

        std::cout << std::fixed << std::setprecision(6);
        for (size_t i = 0; i < nabla_w.size(); ++i) {
            std::cout << "Layer " << i << " Weight Gradients:\n" << nabla_w[i] << "\n";
            std::cout << "Layer " << i << " Bias Gradients: " << nabla_b[i].transpose() << "\n";
        }
        std::cout << "Test: " << contextName << " Passed\n\n";
    }

    std::cout << "Test: Compare CPU and GPU gradients\n";
    bool passed = true;
    for (size_t i = 0; i < sizes.size() - 1; ++i) {
        errorMsg = "CPU and GPU weight gradients differ in layer " + std::to_string(i);
        for (int r = 0; r < nabla_w_map["CPUContext"][i].rows(); ++r) {
            for (int c = 0; c < nabla_w_map["CPUContext"][i].cols(); ++c) {
                double diff = std::abs(nabla_w_map["CPUContext"][i](r, c) - nabla_w_map["GPUContext"][i](r, c));
                if (diff > TOL) {
                    std::cerr << "CPU-GPU weight gradient mismatch in layer " << i << " at (" << r << "," << c << "): "
                        << nabla_w_map["CPUContext"][i](r, c) << " vs "
                        << nabla_w_map["GPUContext"][i](r, c) << " (diff = " << diff << ")\n";
                    passed = false;
                    assertApprox(nabla_w_map["CPUContext"][i](r, c), nabla_w_map["GPUContext"][i](r, c), TOL, errorMsg, __FILE__, __LINE__);
                }
            }
        }
        errorMsg = "CPU and GPU bias gradients differ in layer " + std::to_string(i);
        for (int j = 0; j < nabla_b_map["CPUContext"][i].size(); ++j) {
            double diff = std::abs(nabla_b_map["CPUContext"][i](j) - nabla_b_map["GPUContext"][i](j));
            if (diff > TOL) {
                std::cerr << "CPU-GPU bias gradient mismatch in layer " << i << " at index " << j << ": "
                    << nabla_b_map["CPUContext"][i](j) << " vs "
                    << nabla_b_map["GPUContext"][i](j) << " (diff = " << diff << ")\n";
                passed = false;
                assertApprox(nabla_b_map["CPUContext"][i](j), nabla_b_map["GPUContext"][i](j), TOL, errorMsg, __FILE__, __LINE__);
            }
        }
    }

    std::cout << "Test: Compare CPU and GPU " << (passed ? "Passed" : "Failed") << "\n\n";

    if (!passed) {
        all_passed = false;
        std::cerr << "Test: Compare CPU and GPU failed\n";
    }

    if (all_passed) {
        ++passed_tests_;
        std::cout << "----- testBackpropGradientComputation Passed -----\n\n";
    }
    else {
        std::cout << "----- testBackpropGradientComputation Failed -----\n\n";
    }
    return all_passed;
}


void NeuralNetworkTest::testNetworkConstructor()
{
    std::cout << "Running testNetworkConstructor... ";
    ++total_tests_;
    CPUComputationContext cpuContext;
    Network net(network_sizes_, 0.0, Network::LossType::MSE, Network::NeuronType::SIGMOID, &cpuContext);

    //assertTrue(net.evaluate({}, 0).first == 0, "Evaluate on empty data should return 0", __FILE__, __LINE__);
    assertTrue(net.evaluate(std::vector<std::pair<Eigen::VectorXd, int>>{}, 0).first == 0,
        "Evaluate on empty data should return 0", __FILE__, __LINE__);

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
    net.SGD(training_data, 1000, 1, 1.0, &test_data);
    int final_correct = net.evaluate(test_data, test_data.size()).first;

    assertTrue(final_correct >= initial_correct, "SGD did not improve accuracy", __FILE__, __LINE__);
   
    ++passed_tests_;
    std::cout << "Passed" << std::endl;

    // Note: Future test cases (e.g., AND, OR, MNIST) can be added by extending
    // generateXORLikeDataset or creating new dataset generation functions.
}

// Numerical Gradient Checking:
// Compute numerical gradients using the finite difference method:
// ie. dc/dw = C(w+e) - C(w-e) / 2e, 
// where C is the cost (MSE + L2 term) and e or epsilon is a small perturbation
// Compare numerical gradients to analytical gradients from backprop.
void NeuralNetworkTest::testNetworkGradientChecking()
{
    std::cout << "Running testNetworkGradientChecking... ";
    ++total_tests_;

    // Set up small network with L2 regularization
    double lambda = 0.1; // Non-zero to test L2 term
    Network net(network_sizes_, lambda);
    size_t n = 1; // Single example for simplicity
    const double epsilon = 1e-7; // Perturbation for numerical gradient

    // Input and target
    Eigen::VectorXd input(network_sizes_[0]);
    input.setConstant(1.0);
    Eigen::VectorXd target(network_sizes_.back());
    target.setZero();
    target(0) = 1.0;

    // Get analytical gradients
    auto [nabla_b, nabla_w] = net.backprop(input, target, n);

    // Helper to compute cost (MSE + L2)
    auto compute_cost = [&](const Eigen::VectorXd& output) {
        double mse = 0.5 * (output - target).squaredNorm();
        double l2 = 0.0;
        if (lambda > 0.0 && n > 0) {
            for (const auto& layer : net.get_layers()) { // Assuming get_layers() exists
                l2 += layer->get_weights().squaredNorm();
            }
            l2 *= 0.5 * lambda / n;
        }
        return mse + l2;
    };

    // Check gradients for each layer (limit to first few weights/biases for speed)
    for (size_t l = 0; l < nabla_w.size(); ++l) {

        const auto& layer = net.get_layers()[l];
        Eigen::MatrixXd weights = layer->get_weights();
        Eigen::VectorXd biases = layer->get_biases();
        int max_rows = std::min(2, static_cast<int>(weights.rows())); // Limit for speed
        int max_cols = std::min(2, static_cast<int>(weights.cols()));

        // Check weight gradients
        for (int i = 0; i < max_rows; ++i) {
            for (int j = 0; j < max_cols; ++j) {
                // Perturb weight positively
                weights(i, j) += epsilon;
                net.set_layer_weights(l, weights);
                auto output_plus = net.feedforward(input);
                double cost_plus = compute_cost(output_plus);
                
                // Perturb weight negatively
                weights(i, j) -= 2 * epsilon; // Subtract 2*epsilon to go from +epsilon to -epsilon
                net.set_layer_weights(l, weights);
                auto output_minus = net.feedforward(input);
                double cost_minus = compute_cost(output_minus);

                // Restore original weight
                weights(i, j) += epsilon;
                net.set_layer_weights(l, weights);

                // Numerical gradient
                double numerical_grad = (cost_plus - cost_minus) / (2 * epsilon);
                assertApprox(numerical_grad, nabla_w[l](i, j), TOL,
                    "Weight gradient incorrect in layer " + std::to_string(l), __FILE__, __LINE__);
            }
        }

        // Check bias gradients
        for (int i = 0; i < max_rows; ++i) {
            // Perturb bias positively
            biases(i) += epsilon;
            net.set_layer_biases(l, biases);
            auto output_plus = net.feedforward(input);
            double cost_plus = compute_cost(output_plus);

            // Perturb bias negatively
            biases(i) -= 2 * epsilon;
            net.set_layer_biases(l, biases);
            auto output_minus = net.feedforward(input);
            double cost_minus = compute_cost(output_minus);

            // Restore original bias
            biases(i) += epsilon;
            net.set_layer_biases(l, biases);

            // Numerical gradient
            double numerical_grad = (cost_plus - cost_minus) / (2 * epsilon);
            assertApprox(numerical_grad, nabla_b[l](i), TOL,
                "Bias gradient incorrect in layer " + std::to_string(l), __FILE__, __LINE__);
        }
    }

    ++passed_tests_;
    std::cout << "Passed" << std::endl;
}


void NeuralNetworkTest::testComputationContexts()
{
    std::cout << "Running testComputationContexts... ";
    ++total_tests_;

    // Define XOR dataset
    std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> training_data = {
       {Eigen::VectorXd::Zero(2), Eigen::VectorXd::Zero(2)},
       {Eigen::VectorXd::Unit(2, 0), Eigen::VectorXd::Unit(2, 1)},
       {Eigen::VectorXd::Unit(2, 1), Eigen::VectorXd::Unit(2, 1)},
       {Eigen::VectorXd::Ones(2), Eigen::VectorXd::Zero(2)}
    };

    // Define network architecture
    std::vector<int> sizes = { 2, 3, 2 };

    // Create CPU and GPU computation contexts
    CPUComputationContext cpuContext;
    GPUComputationContext gpuContext;

    // Create networks with CPU and GPU contexts
    Network netCPU(sizes, 0.0, Network::LossType::MSE, Network::NeuronType::SIGMOID, &cpuContext);
    Network netGPU(sizes, 0.0, Network::LossType::MSE, Network::NeuronType::SIGMOID, &gpuContext);

    // Train both networks
    netCPU.SGD(training_data, 1000, 1, 0.1, nullptr, false);
    netGPU.SGD(training_data, 1000, 1, 0.1, nullptr, false);

    // Evaluate both networks
    double cpuLoss = netCPU.evaluate(training_data, training_data.size()).second;
    double gpuLoss = netGPU.evaluate(training_data, training_data.size()).second;

    // Check if losses are approximately equal
    assertApprox(cpuLoss, gpuLoss, TOL, "Losses differ between CPU and GPU", __FILE__, __LINE__);

    // Check predictions
    for (const auto& data : training_data) {
        Eigen::VectorXd cpuOutput = netCPU.feedforward(data.first);
        Eigen::VectorXd gpuOutput = netGPU.feedforward(data.first);
        for (int i = 0; i < cpuOutput.size(); ++i) {
            assertApprox(cpuOutput(i), gpuOutput(i), TOL, "Outputs differ between CPU and GPU", __FILE__, __LINE__);
        }
    }

    ++passed_tests_;
    std::cout << "Passed" << std::endl;
}


bool NeuralNetworkTest::testEvaluate() {
    std::cout << "----- Running testEvaluateCrossEntropy... -----\n";
    ++total_tests_;
    bool all_passed = true;
    std::string contextName;
    std::string errorMsg;

    std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> test_data = {
        {Eigen::VectorXd::Zero(2), Eigen::VectorXd::Zero(2)},
        {Eigen::VectorXd::Unit(2, 0), Eigen::VectorXd::Unit(2, 1)}
    };
    std::vector<int> sizes = { 2, 3, 2 };
    unsigned int seed = 42;
    size_t n = 2;

    double cpu_loss, gpu_loss;
    int cpu_correct, gpu_correct;

    for (auto context : computeContexts) {
        if (dynamic_cast<CPUComputationContext*>(context)) {
            contextName = "CPUContext";
        }
        else if (dynamic_cast<GPUComputationContext*>(context)) {
            contextName = "GPUContext";
        }
        else {
            throw std::runtime_error("Unknown computation context");
        }

        std::cout << "Test: " << contextName << "\n";
        Network net(sizes, 0.0, Network::LossType::CROSS_ENTROPY, Network::NeuronType::SIGMOID, context, seed);
        auto [correct, loss] = net.evaluate(test_data, n);

        if (contextName == "CPUContext") {
            cpu_correct = correct;
            cpu_loss = loss;
        }
        else {
            gpu_correct = correct;
            gpu_loss = loss;
        }

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Correct: " << correct << "/" << test_data.size() << "\n";
        std::cout << "Loss: " << loss << "\n";
        std::cout << "Test: " << contextName << " Passed\n\n";
    }

    std::cout << "Test: Compare CPU and GPU results\n";
    bool passed = true;
    errorMsg = "CPU and GPU correct predictions differ";
    if (cpu_correct != gpu_correct) {
        passed = false;
        std::cerr << errorMsg << ": CPU=" << cpu_correct << ", GPU=" << gpu_correct << "\n";
    }
    errorMsg = "CPU and GPU losses differ";
    if (std::abs(cpu_loss - gpu_loss) > TOL) {
        passed = false;
        std::cerr << errorMsg << ": CPU=" << cpu_loss << ", GPU=" << gpu_loss << " (diff=" << std::abs(cpu_loss - gpu_loss) << ")\n";
    }

    std::cout << "Test: Compare CPU and GPU " << (passed ? "Passed" : "Failed") << "\n\n";

    if (passed) {
        ++passed_tests_;
        std::cout << "----- testEvaluateCrossEntropy Passed -----\n\n";
    }
    else {
        std::cout << "----- testEvaluateCrossEntropy Failed -----\n\n";
    }
    return passed;
}

bool NeuralNetworkTest::runAllTests()
{
    passed_tests_ = 0;
    total_tests_ = 0;
    //testLayerComputeActivationDerivative();
    //testLayerConstructor();
    testLayerForward();
    //testLayerGradients();
    //testLayerUpdateParameters();
    //testNetworkConstructor();
    //testNetworkFeedforward();
    //testNetworkBackprop();
    //testNetworkSGD();
    //testNetworkGradientChecking();
    //testComputationContexts();
    //testUpdateMiniBatchSimplified(); 
    //testBackpropGradientComputation();
    //testEvaluate();
    //doStuff();
    //testLayerComputeActivationDerivativeGPU();
    std::cout << "Test Summary: " << passed_tests_ << "/" << total_tests_ << " tests passed" << std::endl;
    return passed_tests_ == total_tests_;
}

void NeuralNetworkTest::doStuff()
{
    std::cout << "Running doStuff... ";
    std::string contextName;
    std::string errorMsg;

    // Define mini-batch (XOR-like dataset subset)
    std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> mini_batch = {
        {Eigen::VectorXd::Zero(2), Eigen::VectorXd::Zero(2)},
        {Eigen::VectorXd::Unit(2, 0), Eigen::VectorXd::Unit(2, 1)}
    };
    double eta = 0.1;
    size_t n = 2;
    unsigned int seed = 42; // Fixed seed for reproducibility

    // Network configuration
    std::vector<int> sizes = { 2, 3, 2 };
    std::map<std::string, std::vector<Eigen::MatrixXd>> weights_map;
    std::map<std::string, std::vector<Eigen::VectorXd>> biases_map;

    // Test CPU and GPU contexts
    //for (auto context : computeContexts) {
    //    if (dynamic_cast<CPUComputationContext*>(context)) {
    //        contextName = "CPUContext";
    //    }
    //    else if (dynamic_cast<GPUComputationContext*>(context)) {
    //        contextName = "GPUContext";
    //    }
    //    else {
    //        std::cerr << "Unknown computation context in testUpdateMiniBatchSimplified\n";
    //        throw std::runtime_error("Unknown computation context");
    //    }

    //    std::cout << "\nTest: " << contextName << "\n";
    //    Network net(sizes, 0.0, Network::LossType::MSE, Network::NeuronType::SIGMOID, context, seed);

    //    // Run update_mini_batch
    //    double norm = net.update_mini_batch(mini_batch, eta, n);

    //}

    std::cout << "\nTest: GpuCOntext " << "\n";
    GPUComputationContext gpuContext;
    CPUComputationContext cpuContext;
    Network net(sizes, 0.0, Network::LossType::MSE, Network::NeuronType::SIGMOID, &gpuContext, seed);
    double norm = net.update_mini_batch(mini_batch, eta, n);
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

