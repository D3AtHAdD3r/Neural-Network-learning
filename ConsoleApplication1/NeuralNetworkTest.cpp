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

void NeuralNetworkTest::getPrecomputedBackpropTestData(Network::LossType loss_type,
    std::vector<Eigen::MatrixXd>& weights,
    std::vector<Eigen::VectorXd>& biases,
    Eigen::VectorXd& x,
    Eigen::VectorXd& y,
    std::vector<Eigen::VectorXd>& expected_nabla_b,
    std::vector<Eigen::MatrixXd>& expected_nabla_w) const {

    // Set fixed weights and biases for the 2-3-2 network
    // Beginner note: These are predefined values for reproducibility, allowing us to compare against manually computed gradients.
    weights.resize(2);
    weights[0] = Eigen::MatrixXd(3, 2);
    weights[0] << 0.1, 0.2,
        0.3, 0.4,
        0.5, 0.6;
    weights[1] = Eigen::MatrixXd(2, 3);
    weights[1] << 0.7, 0.8, 0.9,
        1.0, 1.1, 1.2;

    biases.resize(2);
    biases[0] = Eigen::VectorXd(3);
    biases[0] << 0.1, 0.2, 0.3;
    biases[1] = Eigen::VectorXd(2);
    biases[1] << 0.4, 0.5;

    // Fixed input and target
    // Beginner note: This is a single example for simplicity in manual verification.
    x = Eigen::VectorXd(2);
    x << 0.5, 0.5;
    y = Eigen::VectorXd(2);
    y << 1.0, 0.0;

    // Precomputed expected gradients based on loss type
    // Beginner note: These values were calculated using Python/NumPy to simulate the backpropagation steps exactly.
    expected_nabla_b.resize(2);
    expected_nabla_w.resize(2);

    if (loss_type == Network::LossType::MSE) {
        expected_nabla_b[0] = Eigen::VectorXd(3);
        expected_nabla_b[0] << 0.01232889, 0.01268599, 0.01243287;

        expected_nabla_w[0] = Eigen::MatrixXd(3, 2);
        expected_nabla_w[0] << 0.00616445, 0.00616445,
            0.006343, 0.006343,
            0.00621643, 0.00621643;

        expected_nabla_b[1] = Eigen::VectorXd(2);
        expected_nabla_b[1] << -0.01399889, 0.05988938;

        expected_nabla_w[1] = Eigen::MatrixXd(2, 3);
        expected_nabla_w[1] << -0.00786985, -0.0088772, -0.00980717,
            0.0336684, 0.03797799, 0.04195653;
    }
    else { // CROSS_ENTROPY
        expected_nabla_b[0] = Eigen::VectorXd(3);
        expected_nabla_b[0] << 0.2073104, 0.21407222, 0.21042799;

        expected_nabla_w[0] = Eigen::MatrixXd(3, 2);
        expected_nabla_w[0] << 0.1036552, 0.1036552,
            0.10703611, 0.10703611,
            0.105214, 0.105214;

        expected_nabla_b[1] = Eigen::VectorXd(2);
        expected_nabla_b[1] << -0.12660205, 0.93088757;

        expected_nabla_w[1] = Eigen::MatrixXd(2, 3);
        expected_nabla_w[1] << -0.0711727, -0.08028287, -0.08869324,
            0.52332312, 0.59030894, 0.65214925;
    }

}


bool NeuralNetworkTest::testNetworkBackprop() {
    // Test 1: Backpropagation on CPU matches manual calculations
    // Beginner note: This test verifies that the CPU version of backpropagation produces gradients that match pre-computed manual values for a small network.
    std::cout << "Running Test 1: Backprop CPU vs Manual" << std::endl;
    total_tests_++;
    {
        // Retrieve precomputed data for MSE (you can change to CROSS_ENTROPY to test the other loss)
        std::vector<Eigen::MatrixXd> weights;
        std::vector<Eigen::VectorXd> biases;
        Eigen::VectorXd x;
        Eigen::VectorXd y;
        std::vector<Eigen::VectorXd> expected_nabla_b;
        std::vector<Eigen::MatrixXd> expected_nabla_w;
        getPrecomputedBackpropTestData(Network::LossType::MSE, weights, biases, x, y, expected_nabla_b, expected_nabla_w);
        
        // Create network with CPU context, no regularization, specified loss, sigmoid neurons
        Network net_cpu(network_sizes_, 0.0, Network::LossType::CROSS_ENTROPY, neuron_type_, cpuContext.get(), seed_);

        // Set precomputed weights and biases
        for (size_t i = 0; i < weights.size(); ++i) {
            net_cpu.set_layer_weights(i, weights[i]);
            net_cpu.set_layer_biases(i, biases[i]);
        }
    
        // Run backpropagation on CPU
        auto [nabla_b_cpu, nabla_w_cpu] = net_cpu.backprop_cpu(x, y, 1);  // n=1 since single example, no regularization impact

        // Assert approximations
        // Beginner note: We check each layer's bias and weight gradients separately for clarity.
        assertVectorApprox(nabla_b_cpu[0], expected_nabla_b[0], TOL, "Hidden layer bias gradients mismatch", __FILE__, __LINE__);
        assertMatrixApprox(nabla_w_cpu[0], expected_nabla_w[0], TOL, "Hidden layer weight gradients mismatch", __FILE__, __LINE__);
        assertVectorApprox(nabla_b_cpu[1], expected_nabla_b[1], TOL, "Output layer bias gradients mismatch", __FILE__, __LINE__);
        assertMatrixApprox(nabla_w_cpu[1], expected_nabla_w[1], TOL, "Output layer weight gradients mismatch", __FILE__, __LINE__);

        std::cout << "Test 1: Backprop CPU vs Manual Passed." << std::endl;
        passed_tests_++;
    }

    // Test 2: Backpropagation on GPU matches manual calculations
    // Beginner note: Similar to Test 1, but for GPU. We copy gradients back from device memory to compare.
    std::cout << "Running Test 2: Backprop GPU vs Manual" << std::endl;
    total_tests_++;
    {
        // Retrieve precomputed data for MSE (you can change to CROSS_ENTROPY to test the other loss)
        std::vector<Eigen::MatrixXd> weights;
        std::vector<Eigen::VectorXd> biases;
        Eigen::VectorXd x;
        Eigen::VectorXd y;
        std::vector<Eigen::VectorXd> expected_nabla_b;
        std::vector<Eigen::MatrixXd> expected_nabla_w;
        getPrecomputedBackpropTestData(Network::LossType::MSE, weights, biases, x, y, expected_nabla_b, expected_nabla_w);

        // Create network with GPU context
        Network net_gpu(network_sizes_, 0.0, Network::LossType::MSE, neuron_type_, gpuContext.get(), seed_);

        // Set precomputed weights and biases
        for (size_t i = 0; i < weights.size(); ++i) {
            net_gpu.set_layer_weights(i, weights[i]);
            net_gpu.set_layer_biases(i, biases[i]);
        }

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

        // Assert
        assertVectorApprox(nabla_b0_gpu, expected_nabla_b[0], TOL, "GPU Hidden layer bias gradients mismatch", __FILE__, __LINE__);
        assertMatrixApprox(nabla_w0_gpu, expected_nabla_w[0], TOL, "GPU Hidden layer weight gradients mismatch", __FILE__, __LINE__);
        assertVectorApprox(nabla_b1_gpu, expected_nabla_b[1], TOL, "GPU Output layer bias gradients mismatch", __FILE__, __LINE__);
        assertMatrixApprox(nabla_w1_gpu, expected_nabla_w[1], TOL, "GPU Output layer weight gradients mismatch", __FILE__, __LINE__);

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
        Eigen::VectorXd nabla_b0_gpu(3);
        gpuContext->copy_biases_to_host(nabla_b0_gpu, gpu_const_layers[0]->get_d_delta_(), 3);
        Eigen::MatrixXd nabla_w1_gpu(2, 3);
        gpuContext->copy_weights_to_host(nabla_w1_gpu, gpu_const_layers[1]->get_d_weight_grads_(), 2, 3);
        Eigen::VectorXd nabla_b1_gpu(2);
        gpuContext->copy_biases_to_host(nabla_b1_gpu, gpu_const_layers[1]->get_d_delta_(), 2);

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


bool NeuralNetworkTest::testUpdateMiniBatch() {
    // Test 4: update_mini_batch on CPU with mini-batch size 1 and lambda=0 matches manual update
    // Beginner note: This test verifies that update_mini_batch correctly applies gradients from a single example 
    // (using backprop) to update weights and biases on CPU, without regularization.

    std::cout << "Running Test 4: update_mini_batch CPU (size=1, lambda=0) vs Manual" << std::endl;
    total_tests_++;
    {
        // Retrieve precomputed data for MSE
        std::vector<Eigen::MatrixXd> weights;
        std::vector<Eigen::VectorXd> biases;
        Eigen::VectorXd x;
        Eigen::VectorXd y;
        std::vector<Eigen::VectorXd> expected_nabla_b;
        std::vector<Eigen::MatrixXd> expected_nabla_w;
        getPrecomputedBackpropTestData(Network::LossType::MSE, weights, biases, x, y, expected_nabla_b, expected_nabla_w);

        // Create network with CPU context, lambda=0, MSE loss
        Network net_cpu(network_sizes_, 0.0, Network::LossType::MSE, neuron_type_, cpuContext.get(), seed_);

        // Set precomputed weights and biases
        for (size_t i = 0; i < weights.size(); ++i) {
            net_cpu.set_layer_weights(i, weights[i]);
            net_cpu.set_layer_biases(i, biases[i]);
        }

        // Create mini-batch with single example
        std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> mini_batch = { {x, y} };

        // Choose a learning rate (eta)
        double eta = 0.1;
        size_t n = 1;  // Total training size (irrelevant for lambda=0)

        // Call update_mini_batch
        net_cpu.update_mini_batch(mini_batch, eta, n);

        // Compute expected updated weights and biases
        // Beginner note: For mini-batch size=1 and lambda=0, update is: param -= eta * gradient
        std::vector<Eigen::MatrixXd> expected_new_weights(2);
        std::vector<Eigen::VectorXd> expected_new_biases(2);
        expected_new_weights[0] = weights[0] - eta * expected_nabla_w[0];
        expected_new_biases[0] = biases[0] - eta * expected_nabla_b[0];
        expected_new_weights[1] = weights[1] - eta * expected_nabla_w[1];
        expected_new_biases[1] = biases[1] - eta * expected_nabla_b[1];

        // Get actual updated parameters from network
        const auto& layers = net_cpu.get_layers();
        assertMatrixApprox(layers[0]->get_weights(), expected_new_weights[0], TOL, "Hidden layer weights after update mismatch", __FILE__, __LINE__);
        assertVectorApprox(layers[0]->get_biases(), expected_new_biases[0], TOL, "Hidden layer biases after update mismatch", __FILE__, __LINE__);
        assertMatrixApprox(layers[1]->get_weights(), expected_new_weights[1], TOL, "Output layer weights after update mismatch", __FILE__, __LINE__);
        assertVectorApprox(layers[1]->get_biases(), expected_new_biases[1], TOL, "Output layer biases after update mismatch", __FILE__, __LINE__);

        std::cout << "Test 4: update_mini_batch CPU (size=1, lambda=0) vs Manual Passed." << std::endl;
        passed_tests_++;
    }

    // Test 5: update_mini_batch on GPU with mini-batch size 1 and lambda=0 matches manual update
    // Beginner note: Similar to Test 4, but for GPU. We copy updated parameters back from device to compare.
    std::cout << "Running Test 5: update_mini_batch GPU (size=1, lambda=0) vs Manual" << std::endl;
    total_tests_++;
    {
        // Retrieve precomputed data for MSE
        std::vector<Eigen::MatrixXd> weights;
        std::vector<Eigen::VectorXd> biases;
        Eigen::VectorXd x;
        Eigen::VectorXd y;
        std::vector<Eigen::VectorXd> expected_nabla_b;
        std::vector<Eigen::MatrixXd> expected_nabla_w;
        getPrecomputedBackpropTestData(Network::LossType::MSE, weights, biases, x, y, expected_nabla_b, expected_nabla_w);

        // Create network with GPU context, lambda=0, MSE loss
        Network net_gpu(network_sizes_, 0.0, Network::LossType::MSE, neuron_type_, gpuContext.get(), seed_);

        // Set precomputed weights and biases
        for (size_t i = 0; i < weights.size(); ++i) {
            net_gpu.set_layer_weights(i, weights[i]);
            net_gpu.set_layer_biases(i, biases[i]);
        }

        // Create mini-batch with single example
        std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> mini_batch = { {x, y} };

        // Choose a learning rate (eta)
        double eta = 0.1;
        size_t n = 1;  // Total training size (irrelevant for lambda=0)

        // Call update_mini_batch
        net_gpu.update_mini_batch(mini_batch, eta, n);

        // Compute expected updated weights and biases
        std::vector<Eigen::MatrixXd> expected_new_weights(2);
        std::vector<Eigen::VectorXd> expected_new_biases(2);
        expected_new_weights[0] = weights[0] - eta * expected_nabla_w[0];
        expected_new_biases[0] = biases[0] - eta * expected_nabla_b[0];
        expected_new_weights[1] = weights[1] - eta * expected_nabla_w[1];
        expected_new_biases[1] = biases[1] - eta * expected_nabla_b[1];

        // Retrieve updated parameters from GPU layers by copying back to host
        // Beginner note: GPU stores parameters in device memory; we copy them to Eigen for comparison.
        const auto& layers = net_gpu.get_layers();

        // Hidden layer
        Eigen::MatrixXd new_weights0_gpu(3, 2);
        gpuContext->copy_weights_to_host(new_weights0_gpu, layers[0]->get_d_weights(), 3, 2);
        Eigen::VectorXd new_biases0_gpu(3);
        gpuContext->copy_biases_to_host(new_biases0_gpu, layers[0]->get_d_biases(), 3);

        // Output layer
        Eigen::MatrixXd new_weights1_gpu(2, 3);
        gpuContext->copy_weights_to_host(new_weights1_gpu, layers[1]->get_d_weights(), 2, 3);
        Eigen::VectorXd new_biases1_gpu(2);
        gpuContext->copy_biases_to_host(new_biases1_gpu, layers[1]->get_d_biases(), 2);

        // Assert
        assertMatrixApprox(new_weights0_gpu, expected_new_weights[0], TOL, "GPU Hidden layer weights after update mismatch", __FILE__, __LINE__);
        assertVectorApprox(new_biases0_gpu, expected_new_biases[0], TOL, "GPU Hidden layer biases after update mismatch", __FILE__, __LINE__);
        assertMatrixApprox(new_weights1_gpu, expected_new_weights[1], TOL, "GPU Output layer weights after update mismatch", __FILE__, __LINE__);
        assertVectorApprox(new_biases1_gpu, expected_new_biases[1], TOL, "GPU Output layer biases after update mismatch", __FILE__, __LINE__);

        std::cout << "Test 5: update_mini_batch GPU (size=1, lambda=0) vs Manual Passed" << std::endl;
        passed_tests_++;
    }

    // Test 6: update_mini_batch consistency between CPU and GPU with mini-batch size >1 and lambda>0
    // Beginner note: This test uses a small dataset (XOR-like) to check if CPU and GPU produce the same updates 
    // when processing multiple examples and applying L2 regularization.
    std::cout << "Running Test 6: update_mini_batch CPU vs GPU Consistency (size>1, lambda>0)" << std::endl;
    total_tests_++;
    {
        // Set lambda >0 for regularization test
        double lambda = 0.01;

        // Create CPU and GPU networks with same seed and lambda>0
        Network net_cpu(network_sizes_, lambda, Network::LossType::MSE, neuron_type_, cpuContext.get(), seed_);
        Network net_gpu(network_sizes_, lambda, Network::LossType::MSE, neuron_type_, gpuContext.get(), seed_);

        // Ensure identical initial parameters by copying from CPU to GPU
        const auto& cpu_layers = net_cpu.get_layers();
        auto& gpu_layers = net_gpu.get_mutable_layers();
        for (size_t i = 0; i < cpu_layers.size(); ++i) {
            gpu_layers[i]->set_weights(cpu_layers[i]->get_weights());
            gpu_layers[i]->set_biases(cpu_layers[i]->get_biases());
        }

        // Generate XOR-like dataset
        std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> training_data;
        std::vector<std::pair<Eigen::VectorXd, int>> test_data;
        generateXORLikeDataset(training_data, test_data);

        // Use the full training_data as mini-batch (size=4)
        auto mini_batch = training_data;

        // Choose eta and n (n=training_data.size() for proper reg scaling)
        double eta = 0.1;
        size_t n = training_data.size();

        // Call update_mini_batch on both
        net_cpu.update_mini_batch(mini_batch, eta, n);
        net_gpu.update_mini_batch(mini_batch, eta, n);

        // Retrieve CPU updated parameters
        Eigen::MatrixXd cpu_new_weights0 = cpu_layers[0]->get_weights();
        Eigen::VectorXd cpu_new_biases0 = cpu_layers[0]->get_biases();
        Eigen::MatrixXd cpu_new_weights1 = cpu_layers[1]->get_weights();
        Eigen::VectorXd cpu_new_biases1 = cpu_layers[1]->get_biases();

        // Retrieve GPU updated parameters
        const auto& gpu_const_layers = net_gpu.get_layers();
        Eigen::MatrixXd gpu_new_weights0(3, 2);
        gpuContext->copy_weights_to_host(gpu_new_weights0, gpu_const_layers[0]->get_d_weights(), 3, 2);
        Eigen::VectorXd gpu_new_biases0(3);
        gpuContext->copy_biases_to_host(gpu_new_biases0, gpu_const_layers[0]->get_d_biases(), 3);
        Eigen::MatrixXd gpu_new_weights1(2, 3);
        gpuContext->copy_weights_to_host(gpu_new_weights1, gpu_const_layers[1]->get_d_weights(), 2, 3);
        Eigen::VectorXd gpu_new_biases1(2);
        gpuContext->copy_biases_to_host(gpu_new_biases1, gpu_const_layers[1]->get_d_biases(), 2);

        // Compare CPU vs GPU updated parameters
        assertMatrixApprox(cpu_new_weights0, gpu_new_weights0, TOL, "Hidden weights CPU vs GPU mismatch after update", __FILE__, __LINE__);
        assertVectorApprox(cpu_new_biases0, gpu_new_biases0, TOL, "Hidden biases CPU vs GPU mismatch after update", __FILE__, __LINE__);
        assertMatrixApprox(cpu_new_weights1, gpu_new_weights1, TOL, "Output weights CPU vs GPU mismatch after update", __FILE__, __LINE__);
        assertVectorApprox(cpu_new_biases1, gpu_new_biases1, TOL, "Output biases CPU vs GPU mismatch after update", __FILE__, __LINE__);

        std::cout << "Test 6: update_mini_batch CPU vs GPU Consistency (size>1, lambda>0) - Passed" << std::endl;
        passed_tests_++;
    }
    return true;
}


bool NeuralNetworkTest::runAllTests()
{
    passed_tests_ = 0;
    total_tests_ = 0;
    
    //tests
    //testNetworkBackprop();
    testUpdateMiniBatch();

    std::cout << "Test Summary: " << passed_tests_ << "/" << total_tests_ << " tests passed" << std::endl;
    return passed_tests_ == total_tests_;
}





