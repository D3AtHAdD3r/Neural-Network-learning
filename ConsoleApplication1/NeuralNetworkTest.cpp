#include "NeuralNetworkTest.hpp"
#include "LegacyFuncs.h"
#include"utils.h"
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

// Helper function to assert vector approximations
// This checks if two Eigen vectors are approximately equal element-wise within a tolerance.
// Beginner note: We use this to compare computed gradients with expected values.
void NeuralNetworkTest::assertVectorApprox(const Eigen::VectorXd& a, const Eigen::VectorXd& b, double tol, const std::string& message, const char* file, int line) {
    assertTrue(a.size() == b.size(), "Vector size mismatch in " + message, file, line);
    for (Eigen::Index i = 0; i < a.size(); ++i) {
        assertApprox(a(i), b(i), tol, message + " at index " + std::to_string(i), file, line);
    }
}

// Helper function to assert matrix approximations
// This checks if two Eigen matrices are approximately equal element-wise within a tolerance.
// Beginner note: Matrices hold weights or gradients; we compare them to ensure calculations match.
void NeuralNetworkTest::assertMatrixApprox(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b, double tol, const std::string& message, const char* file, int line) {
    assertTrue(a.rows() == b.rows() && a.cols() == b.cols(), "Matrix size mismatch in " + message, file, line);
    for (Eigen::Index r = 0; r < a.rows(); ++r) {
        for (Eigen::Index c = 0; c < a.cols(); ++c) {
            assertApprox(a(r, c), b(r, c), tol, message + " at (" + std::to_string(r) + "," + std::to_string(c) + ")", file, line);
        }
    }
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


bool NeuralNetworkTest::testNetworkBackprop() {
    // Test 1: Backpropagation on CPU matches manual calculations
    // Beginner note: This test verifies that the CPU version of backpropagation produces gradients that match pre-computed manual values for a small network.
    std::cout << "Running Test 1: Backprop CPU vs Manual" << std::endl;
    total_tests_++;

    {
        // Create network with CPU context, no regularization, MSE loss, sigmoid neurons
        Network net_cpu(network_sizes_, 0.0, Network::LossType::MSE, neuron_type_, cpuContext.get(), seed_);
    
        // Set specific weights and biases for reproducibility and manual verification
        // Layer 0 (hidden): 3 neurons, 2 inputs
        Eigen::MatrixXd w0(3, 2);
        w0 << 0.1, 0.2,
            0.3, 0.4,
            0.5, 0.6;
        Eigen::VectorXd b0(3);
        b0 << 0.1, 0.2, 0.3;
        net_cpu.set_layer_weights(0, w0);
        net_cpu.set_layer_biases(0, b0);

        // Layer 1 (output): 2 neurons, 3 inputs
        Eigen::MatrixXd w1(2, 3);
        w1 << 0.7, 0.8, 0.9,
            1.0, 1.1, 1.2;
        Eigen::VectorXd b1(2);
        b1 << 0.4, 0.5;
        net_cpu.set_layer_weights(1, w1);
        net_cpu.set_layer_biases(1, b1);

        // Input and target for a single example
        Eigen::VectorXd x(2);
        x << 0.5, 0.5;
        Eigen::VectorXd y(2);
        y << 1.0, 0.0;

        // Run backpropagation on CPU
        auto [nabla_b_cpu, nabla_w_cpu] = net_cpu.backprop_cpu(x, y, 1);  // n=1 since single example, no regularization impact

        // Expected gradients from manual calculation (using Python/numpy for precision)
        // Beginner note: These values were computed step-by-step using the backpropagation formulas for MSE loss and sigmoid activation.
        Eigen::VectorXd expected_b0(3);
        expected_b0 << 0.01232889, 0.01268599, 0.01243287;

        Eigen::MatrixXd expected_w0(3, 2);
        expected_w0 << 0.00616445, 0.00616445,
            0.00634300, 0.00634300,
            0.00621643, 0.00621643;

        Eigen::VectorXd expected_b1(2);
        expected_b1 << -0.01399889, 0.05988938;
        Eigen::MatrixXd expected_w1(2, 3);
        expected_w1 << -0.00786985, -0.00887720, -0.00980717,
            0.03366840, 0.03797799, 0.04195653;

        // Assert approximations
        // Beginner note: We check each layer's bias and weight gradients separately for clarity.
        assertVectorApprox(nabla_b_cpu[0], expected_b0, TOL, "Hidden layer bias gradients mismatch", __FILE__, __LINE__);
        assertMatrixApprox(nabla_w_cpu[0], expected_w0, TOL, "Hidden layer weight gradients mismatch", __FILE__, __LINE__);
        assertVectorApprox(nabla_b_cpu[1], expected_b1, TOL, "Output layer bias gradients mismatch", __FILE__, __LINE__);
        assertMatrixApprox(nabla_w_cpu[1], expected_w1, TOL, "Output layer weight gradients mismatch", __FILE__, __LINE__);

        std::cout << "Test 1: Backprop CPU vs Manual Passed." << std::endl;
        passed_tests_++;
    }

    // Test 2: Backpropagation on GPU matches manual calculations
    // Beginner note: Similar to Test 1, but for GPU. We copy gradients back from device memory to compare.
    std::cout << "Running Test 2: Backprop GPU vs Manual" << std::endl;
    total_tests_++;
    {
        // Create network with GPU context
        Network net_gpu(network_sizes_, 0.0, Network::LossType::MSE, neuron_type_, gpuContext.get(), seed_);
    
        // Set the same specific weights and biases as in CPU test
        Eigen::MatrixXd w0(3, 2);
        w0 << 0.1, 0.2,
            0.3, 0.4,
            0.5, 0.6;
        Eigen::VectorXd b0(3);
        b0 << 0.1, 0.2, 0.3;
        net_gpu.set_layer_weights(0, w0);
        net_gpu.set_layer_biases(0, b0);

        Eigen::MatrixXd w1(2, 3);
        w1 << 0.7, 0.8, 0.9,
            1.0, 1.1, 1.2;
        Eigen::VectorXd b1(2);
        b1 << 0.4, 0.5;
        net_gpu.set_layer_weights(1, w1);
        net_gpu.set_layer_biases(1, b1);

        // Input and target
        Eigen::VectorXd x(2);
        x << 0.5, 0.5;
        Eigen::VectorXd y(2);
        y << 1.0, 0.0;

        // Run backpropagation on GPU (accumulates gradients on device)
        net_gpu.backprop_gpu(x, y, 1);

        // Retrieve gradients from GPU layers by copying back to host
        // Beginner note: GPU computations store results in device memory; we copy them to Eigen structures for comparison.
        const auto& layers = net_gpu.get_layers();

        // Hidden layer
        Eigen::MatrixXd nabla_w0_gpu(3, 2);
        gpuContext->copy_weights_to_host(nabla_w0_gpu, layers[0]->get_d_weight_grads_(), 3, 2);
        Eigen::VectorXd nabla_b0_gpu(3);
        gpuContext->copy_biases_to_host(nabla_b0_gpu, layers[0]->get_d_delta_(), 3);
       
    
        // Output layer
        Eigen::MatrixXd nabla_w1_gpu(2, 3);
        gpuContext->copy_weights_to_host(nabla_w1_gpu, layers[1]->get_d_weight_grads_(), 2, 3);
        Eigen::VectorXd nabla_b1_gpu(2);
        gpuContext->copy_biases_to_host(nabla_b1_gpu, layers[1]->get_d_delta_(), 2);
    
        // Expected values (same as CPU)
        Eigen::VectorXd expected_b0(3);
        expected_b0 << 0.01232889, 0.01268599, 0.01243287;
        Eigen::MatrixXd expected_w0(3, 2);
        expected_w0 << 0.00616445, 0.00616445,
            0.00634300, 0.00634300,
            0.00621643, 0.00621643;
        Eigen::VectorXd expected_b1(2);
        expected_b1 << -0.01399889, 0.05988938;
        Eigen::MatrixXd expected_w1(2, 3);
        expected_w1 << -0.00786985, -0.00887720, -0.00980717,
            0.03366840, 0.03797799, 0.04195653;

        // Assert
        assertVectorApprox(nabla_b0_gpu, expected_b0, TOL, "GPU Hidden layer bias gradients mismatch", __FILE__, __LINE__);
        assertMatrixApprox(nabla_w0_gpu, expected_w0, TOL, "GPU Hidden layer weight gradients mismatch", __FILE__, __LINE__);
        assertVectorApprox(nabla_b1_gpu, expected_b1, TOL, "GPU Output layer bias gradients mismatch", __FILE__, __LINE__);
        assertMatrixApprox(nabla_w1_gpu, expected_w1, TOL, "GPU Output layer weight gradients mismatch", __FILE__, __LINE__);

        std::cout << "Test 2: Backprop GPU vs Manual Passed" << std::endl;
        passed_tests_++;
    }


    // Test 3: Backpropagation consistency between CPU and GPU
    // Beginner note: This test ensures CPU and GPU produce the same gradients by direct comparison, using the XOR-like dataset.
    std::cout << "Running Test 3: Backprop CPU vs GPU Consistency" << std::endl;
    total_tests_++;
    {
        // Create CPU and GPU networks with same seed
        Network net_cpu(network_sizes_, 0.0, Network::LossType::MSE, neuron_type_, cpuContext.get(), seed_);
        Network net_gpu(network_sizes_, 0.0, Network::LossType::MSE, neuron_type_, gpuContext.get(), seed_);

        // Ensure identical initial parameters by copying from CPU to GPU
        // Beginner note: Even with the same seed, we copy to guarantee exact match due to potential RNG differences.
        const auto& cpu_layers = net_cpu.get_layers();
        auto& gpu_layers = net_gpu.get_mutable_layers();
        for (size_t i = 0; i < cpu_layers.size(); ++i) {
            gpu_layers[i]->set_weights(cpu_layers[i]->get_weights());
            gpu_layers[i]->set_biases(cpu_layers[i]->get_biases());
        }

        // Use a sample from XOR-like dataset
        std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> training_data;
        std::vector<std::pair<Eigen::VectorXd, int>> test_data;
        generateXORLikeDataset(training_data, test_data);

        // Pick the first example: input [0,0], target [1,0] (even sum)
        Eigen::VectorXd x = training_data[0].first;
        Eigen::VectorXd y = training_data[0].second;

        // Run backprop on CPU
        auto [nabla_b_cpu, nabla_w_cpu] = net_cpu.backprop_cpu(x, y, training_data.size());

        // Run backprop on GPU
        net_gpu.backprop_gpu(x, y, training_data.size());

        // Retrieve GPU gradients
        const auto& gpu_const_layers = net_gpu.get_layers();  // Use const reference
        Eigen::MatrixXd nabla_w0_gpu(3, 2);
        gpuContext->copy_weights_to_host(nabla_w0_gpu, gpu_const_layers[0]->get_d_weight_grads_(), 3, 2);
        //gpuContext->copy_from_device(gpu_const_layers[0]->get_d_weight_grads_(), nabla_w0_gpu);
        Eigen::VectorXd nabla_b0_gpu(3);
        gpuContext->copy_biases_to_host(nabla_b0_gpu, gpu_const_layers[0]->get_d_delta_(), 3);
        //gpuContext->copy_from_device(gpu_const_layers[0]->get_d_delta_(), nabla_b0_gpu);
        Eigen::MatrixXd nabla_w1_gpu(2, 3);
        gpuContext->copy_weights_to_host(nabla_w1_gpu, gpu_const_layers[1]->get_d_weight_grads_(), 2, 3);
        //gpuContext->copy_from_device(gpu_const_layers[1]->get_d_weight_grads_(), nabla_w1_gpu);
        Eigen::VectorXd nabla_b1_gpu(2);
        gpuContext->copy_biases_to_host(nabla_b1_gpu, gpu_const_layers[1]->get_d_delta_(), 2);
        //gpuContext->copy_from_device(gpu_const_layers[1]->get_d_delta_(), nabla_b1_gpu);
    
        // Compare CPU vs GPU
        // Beginner note: We compare each corresponding gradient to ensure the GPU port matches the CPU implementation.
        assertVectorApprox(nabla_b_cpu[0], nabla_b0_gpu, TOL, "Hidden bias CPU vs GPU mismatch", __FILE__, __LINE__);
        assertMatrixApprox(nabla_w_cpu[0], nabla_w0_gpu, TOL, "Hidden weight CPU vs GPU mismatch", __FILE__, __LINE__);
        assertVectorApprox(nabla_b_cpu[1], nabla_b1_gpu, TOL, "Output bias CPU vs GPU mismatch", __FILE__, __LINE__);
        assertMatrixApprox(nabla_w_cpu[1], nabla_w1_gpu, TOL, "Output weight CPU vs GPU mismatch", __FILE__, __LINE__);

        std::cout << "Test 3: Backprop CPU vs GPU Consistency - Passed" << std::endl;
        passed_tests_++;
    }

    return true;
}


bool NeuralNetworkTest::runAllTests()
{
    passed_tests_ = 0;
    total_tests_ = 0;
    
    //tests
    testNetworkBackprop();

    std::cout << "Test Summary: " << passed_tests_ << "/" << total_tests_ << " tests passed" << std::endl;
    return passed_tests_ == total_tests_;
}





