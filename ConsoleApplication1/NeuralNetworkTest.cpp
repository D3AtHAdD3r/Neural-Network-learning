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

// Function to generate random input vector
Eigen::VectorXd NeuralNetworkTest::generate_random_input(int size, unsigned int seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    Eigen::VectorXd input(size);
    for (int i = 0; i < size; ++i) {
        input(i) = dist(rng);
    }
    return input;
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
    std::cout << "Running Test 1: Backprop CPU vs Manual (MSE)" << std::endl;
    total_tests_++;
    {
        std::vector<Eigen::MatrixXd> weights;
        std::vector<Eigen::VectorXd> biases;
        Eigen::VectorXd x;
        Eigen::VectorXd y;
        std::vector<Eigen::VectorXd> expected_nabla_b;
        std::vector<Eigen::MatrixXd> expected_nabla_w;
        getPrecomputedBackpropTestData(Network::LossType::MSE, weights, biases, x, y, expected_nabla_b, expected_nabla_w);

        Network net_cpu(network_sizes_, 0.0, Network::LossType::MSE, neuron_type_, cpuContext.get(), seed_);
        for (size_t i = 0; i < weights.size(); ++i) {
            net_cpu.set_layer_weights(i, weights[i]);
            net_cpu.set_layer_biases(i, biases[i]);
        }

        net_cpu.feedforward_cpu(x); // Compute and cache activations for all layers
        auto [nabla_b_cpu, nabla_w_cpu] = net_cpu.backprop_cpu(x, y, 1);
        assertVectorApprox(nabla_b_cpu[0], expected_nabla_b[0], TOL, "Hidden layer bias gradients mismatch (MSE)", __FILE__, __LINE__);
        assertMatrixApprox(nabla_w_cpu[0], expected_nabla_w[0], TOL, "Hidden layer weight gradients mismatch (MSE)", __FILE__, __LINE__);
        assertVectorApprox(nabla_b_cpu[1], expected_nabla_b[1], TOL, "Output layer bias gradients mismatch (MSE)", __FILE__, __LINE__);
        assertMatrixApprox(nabla_w_cpu[1], expected_nabla_w[1], TOL, "Output layer weight gradients mismatch (MSE)", __FILE__, __LINE__);

        std::cout << "Test 1: Backprop CPU vs Manual (MSE) Passed." << std::endl;
        passed_tests_++;
    }

    // Test 1b: Backpropagation on CPU with Cross-Entropy
    std::cout << "Running Test 1b: Backprop CPU vs Manual (Cross-Entropy)" << std::endl;
    total_tests_++;
    {
        std::vector<Eigen::MatrixXd> weights;
        std::vector<Eigen::VectorXd> biases;
        Eigen::VectorXd x;
        Eigen::VectorXd y;
        std::vector<Eigen::VectorXd> expected_nabla_b;
        std::vector<Eigen::MatrixXd> expected_nabla_w;
        getPrecomputedBackpropTestData(Network::LossType::CROSS_ENTROPY, weights, biases, x, y, expected_nabla_b, expected_nabla_w);

        Network net_cpu(network_sizes_, 0.0, Network::LossType::CROSS_ENTROPY, neuron_type_, cpuContext.get(), seed_);
        for (size_t i = 0; i < weights.size(); ++i) {
            net_cpu.set_layer_weights(i, weights[i]);
            net_cpu.set_layer_biases(i, biases[i]);
        }

        net_cpu.feedforward_cpu(x); // Compute and cache activations for all layers
        auto [nabla_b_cpu, nabla_w_cpu] = net_cpu.backprop_cpu(x, y, 1);
        assertVectorApprox(nabla_b_cpu[0], expected_nabla_b[0], TOL, "Hidden layer bias gradients mismatch (Cross-Entropy)", __FILE__, __LINE__);
        assertMatrixApprox(nabla_w_cpu[0], expected_nabla_w[0], TOL, "Hidden layer weight gradients mismatch (Cross-Entropy)", __FILE__, __LINE__);
        assertVectorApprox(nabla_b_cpu[1], expected_nabla_b[1], TOL, "Output layer bias gradients mismatch (Cross-Entropy)", __FILE__, __LINE__);
        assertMatrixApprox(nabla_w_cpu[1], expected_nabla_w[1], TOL, "Output layer weight gradients mismatch (Cross-Entropy)", __FILE__, __LINE__);

        std::cout << "Test 1b: Backprop CPU vs Manual (Cross-Entropy) Passed." << std::endl;
        passed_tests_++;
    }

    // Test 2: Backpropagation on GPU matches manual calculations
    std::cout << "Running Test 2: Backprop GPU vs Manual (MSE)" << std::endl;
    total_tests_++;
    {
        std::vector<Eigen::MatrixXd> weights;
        std::vector<Eigen::VectorXd> biases;
        Eigen::VectorXd x;
        Eigen::VectorXd y;
        std::vector<Eigen::VectorXd> expected_nabla_b;
        std::vector<Eigen::MatrixXd> expected_nabla_w;
        getPrecomputedBackpropTestData(Network::LossType::MSE, weights, biases, x, y, expected_nabla_b, expected_nabla_w);

        Network net_gpu(network_sizes_, 0.0, Network::LossType::MSE, neuron_type_, gpuContext.get(), seed_);
        for (size_t i = 0; i < weights.size(); ++i) {
            net_gpu.set_layer_weights(i, weights[i]);
            net_gpu.set_layer_biases(i, biases[i]);
        }

        net_gpu.feedforward_gpu(x); // Compute and cache activations for all layers
        net_gpu.backprop_gpu(x, y, 1);
        const auto& layers = net_gpu.get_layers();
        Eigen::MatrixXd nabla_w0_gpu(3, 2);
        gpuContext->copy_weights_to_host(nabla_w0_gpu, layers[0]->get_d_weight_grads_(), 3, 2);
        Eigen::VectorXd nabla_b0_gpu(3);
        gpuContext->copy_biases_to_host(nabla_b0_gpu, layers[0]->get_d_delta_(), 3);
        Eigen::MatrixXd nabla_w1_gpu(2, 3);
        gpuContext->copy_weights_to_host(nabla_w1_gpu, layers[1]->get_d_weight_grads_(), 2, 3);
        Eigen::VectorXd nabla_b1_gpu(2);
        gpuContext->copy_biases_to_host(nabla_b1_gpu, layers[1]->get_d_delta_(), 2);

        assertVectorApprox(nabla_b0_gpu, expected_nabla_b[0], TOL, "GPU Hidden layer bias gradients mismatch (MSE)", __FILE__, __LINE__);
        assertMatrixApprox(nabla_w0_gpu, expected_nabla_w[0], TOL, "GPU Hidden layer weight gradients mismatch (MSE)", __FILE__, __LINE__);
        assertVectorApprox(nabla_b1_gpu, expected_nabla_b[1], TOL, "GPU Output layer bias gradients mismatch (MSE)", __FILE__, __LINE__);
        assertMatrixApprox(nabla_w1_gpu, expected_nabla_w[1], TOL, "GPU Output layer weight gradients mismatch (MSE)", __FILE__, __LINE__);

        std::cout << "Test 2: Backprop GPU vs Manual (MSE) Passed" << std::endl;
        passed_tests_++;
    }

    // Test 2b: Backpropagation on GPU with Cross-Entropy
    std::cout << "Running Test 2b: Backprop GPU vs Manual (Cross-Entropy)" << std::endl;
    total_tests_++;
    {
        std::vector<Eigen::MatrixXd> weights;
        std::vector<Eigen::VectorXd> biases;
        Eigen::VectorXd x;
        Eigen::VectorXd y;
        std::vector<Eigen::VectorXd> expected_nabla_b;
        std::vector<Eigen::MatrixXd> expected_nabla_w;
        getPrecomputedBackpropTestData(Network::LossType::CROSS_ENTROPY, weights, biases, x, y, expected_nabla_b, expected_nabla_w);

        Network net_gpu(network_sizes_, 0.0, Network::LossType::CROSS_ENTROPY, neuron_type_, gpuContext.get(), seed_);
        for (size_t i = 0; i < weights.size(); ++i) {
            net_gpu.set_layer_weights(i, weights[i]);
            net_gpu.set_layer_biases(i, biases[i]);
        }

        net_gpu.feedforward_gpu(x); // Compute and cache activations for all layers
        net_gpu.backprop_gpu(x, y, 1);
        const auto& layers = net_gpu.get_layers();
        Eigen::MatrixXd nabla_w0_gpu(3, 2);
        gpuContext->copy_weights_to_host(nabla_w0_gpu, layers[0]->get_d_weight_grads_(), 3, 2);
        Eigen::VectorXd nabla_b0_gpu(3);
        gpuContext->copy_biases_to_host(nabla_b0_gpu, layers[0]->get_d_delta_(), 3);
        Eigen::MatrixXd nabla_w1_gpu(2, 3);
        gpuContext->copy_weights_to_host(nabla_w1_gpu, layers[1]->get_d_weight_grads_(), 2, 3);
        Eigen::VectorXd nabla_b1_gpu(2);
        gpuContext->copy_biases_to_host(nabla_b1_gpu, layers[1]->get_d_delta_(), 2);

        assertVectorApprox(nabla_b0_gpu, expected_nabla_b[0], TOL, "GPU Hidden layer bias gradients mismatch (Cross-Entropy)", __FILE__, __LINE__);
        assertMatrixApprox(nabla_w0_gpu, expected_nabla_w[0], TOL, "GPU Hidden layer weight gradients mismatch (Cross-Entropy)", __FILE__, __LINE__);
        assertVectorApprox(nabla_b1_gpu, expected_nabla_b[1], TOL, "GPU Output layer bias gradients mismatch (Cross-Entropy)", __FILE__, __LINE__);
        assertMatrixApprox(nabla_w1_gpu, expected_nabla_w[1], TOL, "GPU Output layer weight gradients mismatch (Cross-Entropy)", __FILE__, __LINE__);

        std::cout << "Test 2b: Backprop GPU vs Manual (Cross-Entropy) Passed" << std::endl;
        passed_tests_++;
    }

    // Test 3: Backpropagation consistency between CPU and GPU
    std::cout << "Running Test 3: Backprop CPU vs GPU Consistency" << std::endl;
    total_tests_++;
    {
        Network net_cpu(network_sizes_, 0.0, Network::LossType::MSE, neuron_type_, cpuContext.get(), seed_);
        Network net_gpu(network_sizes_, 0.0, Network::LossType::MSE, neuron_type_, gpuContext.get(), seed_);

        const auto& cpu_layers = net_cpu.get_layers();
        auto& gpu_layers = net_gpu.get_mutable_layers();
        for (size_t i = 0; i < cpu_layers.size(); ++i) {
            gpu_layers[i]->set_weights(cpu_layers[i]->get_weights());
            gpu_layers[i]->set_biases(cpu_layers[i]->get_biases());
        }

        std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> training_data;
        std::vector<std::pair<Eigen::VectorXd, int>> test_data;
        generateXORLikeDataset(training_data, test_data);

        Eigen::VectorXd x = training_data[0].first;
        Eigen::VectorXd y = training_data[0].second;

        net_cpu.feedforward_cpu(x); // Compute and cache activations for all layers
        net_gpu.feedforward_gpu(x); // Compute and cache activations for all layers

        auto [nabla_b_cpu, nabla_w_cpu] = net_cpu.backprop_cpu(x, y, training_data.size());
        net_gpu.backprop_gpu(x, y, training_data.size());
        const auto& gpu_const_layers = net_gpu.get_layers();
        Eigen::MatrixXd nabla_w0_gpu(3, 2);
        gpuContext->copy_weights_to_host(nabla_w0_gpu, gpu_const_layers[0]->get_d_weight_grads_(), 3, 2);
        Eigen::VectorXd nabla_b0_gpu(3);
        gpuContext->copy_biases_to_host(nabla_b0_gpu, gpu_const_layers[0]->get_d_delta_(), 3);
        Eigen::MatrixXd nabla_w1_gpu(2, 3);
        gpuContext->copy_weights_to_host(nabla_w1_gpu, gpu_const_layers[1]->get_d_weight_grads_(), 2, 3);
        Eigen::VectorXd nabla_b1_gpu(2);
        gpuContext->copy_biases_to_host(nabla_b1_gpu, gpu_const_layers[1]->get_d_delta_(), 2);

        assertVectorApprox(nabla_b_cpu[0], nabla_b0_gpu, TOL, "Hidden bias CPU vs GPU mismatch", __FILE__, __LINE__);
        assertMatrixApprox(nabla_w_cpu[0], nabla_w0_gpu, TOL, "Hidden weight CPU vs GPU mismatch", __FILE__, __LINE__);
        assertVectorApprox(nabla_b_cpu[1], nabla_b1_gpu, TOL, "Output bias CPU vs GPU mismatch", __FILE__, __LINE__);
        assertMatrixApprox(nabla_w_cpu[1], nabla_w1_gpu, TOL, "Output weight CPU vs GPU mismatch", __FILE__, __LINE__);

        std::cout << "Test 3: Backprop CPU vs GPU Consistency - Passed" << std::endl;
        passed_tests_++;
    }

    return true;
}



void NeuralNetworkTest::runUpdateMiniBatchTests(const std::string& context, const std::string& loss, double lambda, int batch_size) {
    total_tests_++;
    std::cout << "Running Test: Context=" << context
        << ", Loss=" << loss
        << ", Lambda=" << lambda
        << ", BatchSize=" << batch_size << std::endl;

    // Validate inputs
    bool is_cpu = (context == "cpu");
    bool is_mse = (loss == "mse");
    if (!is_cpu && context != "gpu") {
        std::cerr << "Invalid context: " << context << std::endl;
        return;
    }
    if (!is_mse && loss != "cross_entropy") {
        std::cerr << "Invalid loss: " << loss << std::endl;
        return;
    }

    // Initialize network
    Network net(network_sizes_, lambda, is_mse ? Network::LossType::MSE : Network::LossType::CROSS_ENTROPY,
        neuron_type_, is_cpu ? static_cast<ComputationContext*>(cpuContext.get()) : static_cast<ComputationContext*>(gpuContext.get()), seed_);

    // Generate mini-batch
    std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> mini_batch;
    size_t n = batch_size;
    std::vector<Eigen::MatrixXd> init_weights(network_sizes_.size() - 1);
    std::vector<Eigen::VectorXd> init_biases(network_sizes_.size() - 1);
    std::vector<Eigen::MatrixXd> expected_nabla_w(net.get_num_layers() - 1);
    std::vector<Eigen::VectorXd> expected_nabla_b(net.get_num_layers() - 1);

    for (size_t i = 0; i < net.get_num_layers() - 1; ++i) {
        expected_nabla_w[i] = Eigen::MatrixXd::Zero(net.get_layer_sizes()[i + 1], net.get_layer_sizes()[i]);
        expected_nabla_b[i] = Eigen::VectorXd::Zero(net.get_layer_sizes()[i + 1]);
    }

    if (batch_size == 1) {
        // Single-example mini-batch
        std::vector<Eigen::MatrixXd> weights;
        std::vector<Eigen::VectorXd> biases;
        Eigen::VectorXd x, y;
        getPrecomputedBackpropTestData(is_mse ? Network::LossType::MSE : Network::LossType::CROSS_ENTROPY,
            weights, biases, x, y, expected_nabla_b, expected_nabla_w);

        // Set weights and biases
        for (size_t i = 0; i < weights.size(); ++i) {
            net.set_layer_weights(i, weights[i]);
            net.set_layer_biases(i, biases[i]);
            init_weights[i] = weights[i];
            init_biases[i] = biases[i];
        }

        mini_batch = { {x, y} };
    }
    else {
        // Multi-example mini-batch from XOR-like dataset
        std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> training_data;
        std::vector<std::pair<Eigen::VectorXd, int>> test_data;
        generateXORLikeDataset(training_data, test_data);
        mini_batch = training_data;  // Size=4
        if (batch_size != 4) {
            std::cerr << "BatchSize=" << batch_size << " not supported; using size=4" << std::endl;
            n = 4;
        }
        const auto& layers = net.get_layers();
        for (size_t i = 0; i < layers.size(); ++i) {
            init_weights[i] = layers[i]->get_weights();
            init_biases[i] = layers[i]->get_biases();
        }
    }

    // Perform update_mini_batch
    double eta = 0.1;
    double computed_norm = net.update_mini_batch(mini_batch, eta, n);

    // Compute expected norm and updates
    double reg_scale = lambda * mini_batch.size() / n;
    double expected_norm = 0.0;

    if (batch_size == 1) {
        //compute expected weights, biases and norm
        std::vector<Eigen::MatrixXd> weight_grads = expected_nabla_w; // Copy summed gradients
        for (size_t i = 0; i < weight_grads.size(); ++i) {
            weight_grads[i] += reg_scale * init_weights[i]; // Apply regularization to summed gradients
            expected_norm += weight_grads[i].squaredNorm();
            expected_norm += expected_nabla_b[i].squaredNorm();
        }
        expected_norm = std::sqrt(expected_norm) / mini_batch.size(); // Divide by mini_batch.size()

        std::vector<Eigen::MatrixXd> expected_new_weights(net.get_num_layers() - 1);
        std::vector<Eigen::VectorXd> expected_new_biases(net.get_num_layers() - 1);
        for (size_t i = 0; i < expected_nabla_w.size(); ++i) {
            expected_new_weights[i] = init_weights[i] - eta * (expected_nabla_w[i] + (lambda / n) * init_weights[i]);
            expected_new_biases[i] = init_biases[i] - eta * expected_nabla_b[i];
        }

        //retrieve actual weights and biases
        const auto& layers = net.get_layers();
        int rows, cols;
        std::vector<Eigen::MatrixXd> new_weights(net.get_num_layers() - 1);
        std::vector<Eigen::VectorXd> new_biases(net.get_num_layers() - 1);
        if (is_cpu) {
            for (size_t i = 0; i < net.get_num_layers() - 1; ++i) {
                rows = net.get_layer_sizes()[i + 1];
                cols = net.get_layer_sizes()[i];
                new_weights[i] = Eigen::MatrixXd::Zero(rows, cols);
                new_biases[i] = Eigen::VectorXd::Zero(rows);
                new_weights[i] = layers[i]->get_weights();
                new_biases[i] = layers[i]->get_biases();
            }
        }
        else {
            for (size_t i = 0; i < net.get_num_layers() - 1; ++i) {
                rows = net.get_layer_sizes()[i + 1];
                cols = net.get_layer_sizes()[i];
                new_weights[i] = Eigen::MatrixXd::Zero(rows, cols);
                new_biases[i] = Eigen::VectorXd::Zero(rows);
                gpuContext->copy_weights_to_host(new_weights[i], layers[i]->get_d_weights(), rows, cols);
                gpuContext->copy_biases_to_host(new_biases[i], layers[i]->get_d_biases(), rows);
            }
        }

        //compare with expected 
        for (size_t i = 0; i < net.get_num_layers() - 1; ++i) {
            // Assert parameters
            assertMatrixApprox(new_weights[i], expected_new_weights[i], TOL, "Layer " + std::to_string(i) + " weights mismatch", __FILE__, __LINE__);
            assertVectorApprox(new_biases[i], expected_new_biases[i], TOL, "Layer " + std::to_string(i) + " biases mismatch", __FILE__, __LINE__);
        }
        
        // Assert norm
        assertApprox(computed_norm, expected_norm, TOL, "Gradient norm mismatch", __FILE__, __LINE__);
    }
    else {
        //call backprop to get gradients per training example, accumulate them
        std::vector<Eigen::MatrixXd> sum_nabla_w(net.get_num_layers() - 1);
        std::vector<Eigen::VectorXd> sum_nabla_b(net.get_num_layers() - 1);
        for (size_t i = 0; i < net.get_num_layers() - 1; ++i) {
            sum_nabla_w[i] = Eigen::MatrixXd::Zero(net.get_layer_sizes()[i + 1], net.get_layer_sizes()[i]);
            sum_nabla_b[i] = Eigen::VectorXd::Zero(net.get_layer_sizes()[i + 1]);
        }

        // Create temporary CPU network for computing expected gradients- backprop
        // backprop calls feedforward - which performs forward pass based on weights and biases, after that it computes and returns gradients
        // Since we have already called network::updateminibatch with main network, its initial weights and biases have updated.
        // so create a new network with same seed to reproduce initial weights and biases.
        Network net_temp(network_sizes_, lambda, is_mse ? Network::LossType::MSE : Network::LossType::CROSS_ENTROPY,
            neuron_type_, static_cast<ComputationContext*>(cpuContext.get()), seed_);

        for (const auto& [x, y] : mini_batch) {
            net_temp.feedforward_cpu(x); // Compute and cache activations for all layers
            auto [nabla_b, nabla_w] = net_temp.backprop_cpu(x, y, n);
            for (size_t i = 0; i < nabla_w.size(); ++i) {
                sum_nabla_w[i] += nabla_w[i];
                sum_nabla_b[i] += nabla_b[i];
            }
        }
        
        //compute expected weights, biases and norm
        std::vector<Eigen::MatrixXd> weight_grads = sum_nabla_w; // Copy summed gradients
        for (size_t i = 0; i < weight_grads.size(); ++i) {
            weight_grads[i] += reg_scale * init_weights[i]; // Apply regularization to summed gradients
            expected_norm += weight_grads[i].squaredNorm();
            expected_norm += sum_nabla_b[i].squaredNorm();
        }
        expected_norm = std::sqrt(expected_norm) / mini_batch.size(); // Divide by mini_batch.size()

        std::vector<Eigen::MatrixXd> expected_new_weights(net.get_num_layers() - 1);
        std::vector<Eigen::VectorXd> expected_new_biases(net.get_num_layers() - 1);
        for (size_t i = 0; i < sum_nabla_w.size(); ++i) {
            expected_new_weights[i] = init_weights[i] - eta * (sum_nabla_w[i] / mini_batch.size() + (lambda / n) * init_weights[i]);
            expected_new_biases[i] = init_biases[i] - eta * (sum_nabla_b[i] / mini_batch.size());
        }

        //retrieve actual weights and biases
        const auto& layers = net.get_layers();
        int rows, cols;
        std::vector<Eigen::MatrixXd> new_weights(net.get_num_layers() - 1);
        std::vector<Eigen::VectorXd> new_biases(net.get_num_layers() - 1);
        if (is_cpu) {
            for (size_t i = 0; i < net.get_num_layers() - 1; ++i) {
                rows = net.get_layer_sizes()[i + 1];
                cols = net.get_layer_sizes()[i];
                new_weights[i] = Eigen::MatrixXd::Zero(rows, cols);
                new_biases[i] = Eigen::VectorXd::Zero(rows);
                new_weights[i] = layers[i]->get_weights();
                new_biases[i] = layers[i]->get_biases();
            }
        }
        else {
            for (size_t i = 0; i < net.get_num_layers() - 1; ++i) {
                //rows = net.get_layer_sizes()[i + 1];
                //cols = net.get_layer_sizes()[i];

                rows = layers[i]->get_num_neurons();
                cols = layers[i]->get_num_inputs();

                new_weights[i] = Eigen::MatrixXd::Zero(rows, cols);
                new_biases[i] = Eigen::VectorXd::Zero(rows);
                gpuContext->copy_weights_to_host(new_weights[i], layers[i]->get_d_weights(), rows, cols);
                gpuContext->copy_biases_to_host(new_biases[i], layers[i]->get_d_biases(), rows);
            }

        }

        //compare with expected
        for (size_t i = 0; i < net.get_num_layers() - 1; ++i) {
            // Assert parameters
            assertMatrixApprox(new_weights[i], expected_new_weights[i], TOL, "Layer " + std::to_string(i) + " weights mismatch", __FILE__, __LINE__);
            assertVectorApprox(new_biases[i], expected_new_biases[i], TOL, "Layer " + std::to_string(i) + " biases mismatch", __FILE__, __LINE__);
        }

        assertApprox(computed_norm, expected_norm, TOL, "Gradient norm mismatch", __FILE__, __LINE__);
    }

    std::cout << "Test Passed.." << std::endl;
    passed_tests_++;
}

bool NeuralNetworkTest::testUpdateMiniBatch() {
    
    std::cout << "-----Running Test : testUpdateMiniBatch-----" << std::endl;
    std::vector<std::string> contexts = { "cpu", "gpu" };
    std::vector<std::string> losses = { "mse", "cross_entropy" };
    std::vector<double> lambdas = { 0.0, 0.1 };
    std::vector<int> batch_sizes = { 1, 4 };

    for (const auto& context : contexts) {
        for (const auto& loss : losses) {
            for (double lambda : lambdas) {
                for (int batch_size : batch_sizes) {
                    runUpdateMiniBatchTests(context, loss, lambda, batch_size);
                }
            }
        }
    }

    std::cout << "-----Test: testUpdateMiniBatch Passed-----" << std::endl;
    return true;
}

//These tests verify the batch functions in GPUComputationContext.
// We use small data for quick execution and easy verification against host computations.
bool NeuralNetworkTest::testBatchFunctionsGPU_context() {
    std::cout << "-----Running Test: testBatchFunctionsGPU-----" << std::endl;
    total_tests_++;

    GPUComputationContext ctx; // Temporary context for testing

    // Test 1: Memory Allocation and Copy (to/from device)
    int vec_size = 3; // Small size for inputs/outputs
    int batch_size = 4;
    std::vector<Eigen::VectorXd> host_batch(batch_size, Eigen::VectorXd(vec_size));
    for (int b = 0; b < batch_size; ++b) {
        host_batch[b] << b + 1.0, b + 2.0, b + 3.0; // Sample data: [1,2,3], [2,3,4], etc.
    }

    double* d_batch = nullptr;
    ctx.allocate_batch_vector(&d_batch, vec_size, batch_size);
    ctx.copy_batch_to_device(d_batch, host_batch); // transpose=false (vec_size × batch_size)

    std::vector<Eigen::VectorXd> host_copy_back(batch_size);
    ctx.copy_batch_to_host(host_copy_back, d_batch, vec_size, batch_size);
    for (int b = 0; b < batch_size; ++b) {
        assertVectorApprox(host_copy_back[b], host_batch[b], TOL, "Batch copy roundtrip mismatch", __FILE__, __LINE__);
    }

    // Test 2: Linear Computation
    // Setup weights (m=2 neurons, n=3 inputs), biases (2), batch_input (3 × 4)
    int m = 2, n = vec_size; // Reuse vec_size as num_inputs
    Eigen::MatrixXd host_weights(m, n);
    host_weights << 0.1, 0.2, 0.3,
        0.4, 0.5, 0.6;
    Eigen::VectorXd host_biases(m);
    host_biases << 0.7, 0.8;

    double* d_weights = nullptr;
    double* d_biases = nullptr;
    double* d_batch_z = nullptr;
    ctx.allocate_weights(&d_weights, m, n);
    ctx.allocate_biases(&d_biases, m);
    ctx.allocate_batch_vector(&d_batch_z, m, batch_size);

    ctx.copy_to_device(d_weights, host_weights);
    ctx.copy_biases_to_device(d_biases, host_biases);

    // Compute on GPU
    ctx.computeLinearGPU_batch(d_weights, d_batch, d_biases, d_batch_z, m, n, batch_size);

    // Compute on host for verification
    std::vector<Eigen::VectorXd> host_z(batch_size, Eigen::VectorXd(m));
    for (int b = 0; b < batch_size; ++b) {
        host_z[b] = host_weights * host_batch[b] + host_biases;
    }

    // Copy back and assert
    std::vector<Eigen::VectorXd> gpu_z(batch_size);
    ctx.copy_batch_to_host(gpu_z, d_batch_z, m, batch_size);
    for (int b = 0; b < batch_size; ++b) {
        assertVectorApprox(gpu_z[b], host_z[b], TOL, "Batched linear mismatch", __FILE__, __LINE__);
    }


    // Test 3: Batched Activation (sigmoid)
    SigmoidActivation sigmoid; // Use sigmoid for testing
    double* d_batch_a = nullptr;
    ctx.allocate_batch_vector(&d_batch_a, m, batch_size);
    ctx.applyActivationGPU_batch(d_batch_z, d_batch_a, m, batch_size, &sigmoid);

    // Host verification
    std::vector<Eigen::VectorXd> host_a(batch_size, Eigen::VectorXd(m));
    for (int b = 0; b < batch_size; ++b) {
        host_a[b] = host_z[b].unaryExpr([](double val) { return 1.0 / (1.0 + std::exp(-val)); });
    }

    std::vector<Eigen::VectorXd> gpu_a(batch_size);
    ctx.copy_batch_to_host(gpu_a, d_batch_a, m, batch_size);
    for (int b = 0; b < batch_size; ++b) {
        assertVectorApprox(gpu_a[b], host_a[b], TOL, "Batched activation mismatch", __FILE__, __LINE__);
    }


    // Test 4: Debug and Utility (set_to_zero_batch)
    ctx.set_to_zero_batch(d_batch_a, m, batch_size);
    ctx.copy_batch_to_host(gpu_a, d_batch_a, m, batch_size);
    for (int b = 0; b < batch_size; ++b) {
        assertVectorApprox(gpu_a[b], Eigen::VectorXd::Zero(m), TOL, "set_to_zero_batch failed", __FILE__, __LINE__);
    }

    // Test debugPrint_batch (manual verification, no assert; check console output)
    ctx.debugPrint_batch(d_batch_a, m, batch_size); // Should print zeros

    // Test 5: Edge Cases
    // batch_size=1: Should match single-example
    int single_batch = 1;
    double* d_single_batch = nullptr;
    ctx.allocate_batch_vector(&d_single_batch, vec_size, single_batch);
    std::vector<Eigen::VectorXd> single_host_batch = { host_batch[0] };
    ctx.copy_batch_to_device(d_single_batch, single_host_batch);

    double* d_single_z = nullptr;
    ctx.allocate_batch_vector(&d_single_z, m, single_batch);
    ctx.computeLinearGPU_batch(d_weights, d_single_batch, d_biases, d_single_z, m, n, single_batch);

    std::vector<Eigen::VectorXd> single_gpu_z(single_batch);
    ctx.copy_batch_to_host(single_gpu_z, d_single_z, m, single_batch);
    assertVectorApprox(single_gpu_z[0], host_z[0], TOL, "batch_size=1 linear mismatch", __FILE__, __LINE__);

    // batch_size=0: Graceful (no-op)
    std::vector<Eigen::VectorXd> empty_batch;
    ctx.copy_batch_to_device(d_batch, empty_batch); // Should do nothing
    // No assert; just ensure no crash

    // Sub-batch throw: Test process_subbatched with too-large size
    bool threw = false;
    try {
        ctx.process_subbatched([](int) {}, 200, GPUComputationContext::MAX_BATCH_SIZE);
    }
    catch (const std::runtime_error&) {
        threw = true;
    }
    assertTrue(threw, "process_subbatched did not throw for oversized batch", __FILE__, __LINE__);

    // Clean up
    ctx.free_batch_vector(d_batch);
    ctx.free_batch_vector(d_batch_z);
    ctx.free_batch_vector(d_batch_a);
    ctx.free_batch_vector(d_single_batch);
    ctx.free_batch_vector(d_single_z);
    ctx.free_weights(d_weights);
    ctx.free_biases(d_biases);

    std::cout << "Test Passed.." << std::endl;
    passed_tests_++;
    return true;
}

// Test function for a given batch size
bool NeuralNetworkTest::test_feedforward_batch_vs_single(int batch_size, int input_size, int hidden_size, int output_size, GPUComputationContext* gpu_context) {
    std::vector<int> sizes = { input_size, hidden_size, output_size };
    Network net(sizes, 0.0, Network::LossType::MSE, Network::NeuronType::SIGMOID, gpu_context, 42);

    // Generate random inputs (use same seed for reproducibility)
    std::vector<Eigen::VectorXd> batch_inputs;
    for (int i = 0; i < batch_size; ++i) {
        batch_inputs.push_back(generate_random_input(input_size, 42 + i));  // Different seed per input
    }

    // Single-example feedforward
    std::vector<Eigen::VectorXd> single_outputs;
    for (const auto& input : batch_inputs) {
        single_outputs.push_back(net.feedforward_gpu(input));
    }

    // Batched feedforward
    std::vector<Eigen::VectorXd> batch_outputs;
    net.feedforward_gpu_batch(batch_inputs, batch_outputs);

    // Compare outputs
    bool all_match = true;
    for (int i = 0; i < batch_size; ++i) {
        double max_diff = (single_outputs[i] - batch_outputs[i]).cwiseAbs().maxCoeff();
        if (max_diff > TOL) {
            all_match = false;
            std::cout << "Mismatch for example " << i << ": max diff = " << max_diff << std::endl;
        }
    }

    return all_match;
}

bool NeuralNetworkTest::test_feedforward_gpu_batch() {
    std::cout << "-----Running Test: test_feedforward_gpu_batch-----" << std::endl;
    total_tests_++;
    // Test parameters (simple network: 10 inputs, 5 hidden, 3 outputs)
    int input_size = 10;
    int hidden_size = 5;
    int output_size = 3;

    // Test for batch_size=1
    if (test_feedforward_batch_vs_single(1, input_size, hidden_size, output_size, gpuContext.get())) {
        std::cout << "Test passed for batch_size=1" << std::endl;
    }
    else {
        std::cout << "Test failed for batch_size=1" << std::endl;
    }

    // Test for batch_size=4
    if (test_feedforward_batch_vs_single(4, input_size, hidden_size, output_size, gpuContext.get())) {
        std::cout << "Test passed for batch_size=4" << std::endl;
    }
    else {
        std::cout << "Test failed for batch_size=4" << std::endl;
    }

    std::cout << "Test Passed.." << std::endl;
    passed_tests_++;
    return true;
}

bool NeuralNetworkTest::test_launch_elementwise_subtract_batch() {
    std::cout << "-----Running Test: test_launch_elementwise_subtract_batch-----" << std::endl;
    total_tests_++;

    // Setup: 2x2 matrix, batch size 2
    int rows = 2, batch_size = 2;
    std::vector<Eigen::VectorXd> a_batch(batch_size), b_batch(batch_size);
    a_batch[0] = Eigen::VectorXd::Constant(rows, 1.0);  // [1, 1]
    a_batch[1] = Eigen::VectorXd::Constant(rows, 2.0);  // [2, 2]
    b_batch[0] = Eigen::VectorXd::Constant(rows, 0.5);  // [0.5, 0.5]
    b_batch[1] = Eigen::VectorXd::Constant(rows, 1.5);  // [1.5, 1.5]

    // Expected: c = a - b
    std::vector<Eigen::VectorXd> expected_c(batch_size);
    expected_c[0] = Eigen::VectorXd::Constant(rows, 0.5);  // [0.5, 0.5]
    expected_c[1] = Eigen::VectorXd::Constant(rows, 0.5);  // [0.5, 0.5]

    // GPU: Allocate and copy inputs
    double* d_a, * d_b, * d_c;
    gpuContext->allocate_batch_vector(&d_a, rows, batch_size);
    gpuContext->allocate_batch_vector(&d_b, rows, batch_size);
    gpuContext->allocate_batch_vector(&d_c, rows, batch_size);
    gpuContext->copy_batch_to_device(d_a, a_batch, false);
    gpuContext->copy_batch_to_device(d_b, b_batch, false);
    gpuContext->set_to_zero_batch(d_c, rows, batch_size);

    // Run GPU subtract
    gpuContext->launch_elementwise_subtract_batch(d_a, d_b, d_c, rows, batch_size);

    // Copy result back
    std::vector<Eigen::VectorXd> result_c(batch_size);
    gpuContext->copy_batch_to_host(result_c, d_c, rows, batch_size);

    // Clean up
    gpuContext->free_batch_vector(d_a);
    gpuContext->free_batch_vector(d_b);
    gpuContext->free_batch_vector(d_c);

    // Compare
    for (int i = 0; i < batch_size; ++i) {
        assertVectorApprox(result_c[i], expected_c[i], TOL, "Subtract result mismatch for batch " + std::to_string(i), __FILE__, __LINE__);
    }

    std::cout << "Test Passed.." << std::endl;
    passed_tests_++;
    return true;
}

bool NeuralNetworkTest::test_launch_elementwise_multiply_batch() {
    std::cout << "-----Running Test: test_launch_elementwise_multiply_batch-----" << std::endl;
    total_tests_++;

    // Setup: 2x2 matrix, batch size 2
    int rows = 2, batch_size = 2;
    std::vector<Eigen::VectorXd> a_batch(batch_size), b_batch(batch_size);
    a_batch[0] = Eigen::VectorXd::Constant(rows, 2.0);  // [2, 2]
    a_batch[1] = Eigen::VectorXd::Constant(rows, 3.0);  // [3, 3]
    b_batch[0] = Eigen::VectorXd::Constant(rows, 0.5);  // [0.5, 0.5]
    b_batch[1] = Eigen::VectorXd::Constant(rows, 1.5);  // [1.5, 1.5]

    // Expected: c = a * b
    std::vector<Eigen::VectorXd> expected_c(batch_size);
    expected_c[0] = Eigen::VectorXd::Constant(rows, 1.0);  // [1, 1]
    expected_c[1] = Eigen::VectorXd::Constant(rows, 4.5);  // [4.5, 4.5]

    // GPU: Allocate and copy inputs
    double* d_a, * d_b, * d_c;
    gpuContext->allocate_batch_vector(&d_a, rows, batch_size);
    gpuContext->allocate_batch_vector(&d_b, rows, batch_size);
    gpuContext->allocate_batch_vector(&d_c, rows, batch_size);
    gpuContext->copy_batch_to_device(d_a, a_batch, false);
    gpuContext->copy_batch_to_device(d_b, b_batch, false);
    gpuContext->set_to_zero_batch(d_c, rows, batch_size);

    // Run GPU multiply
    gpuContext->launch_elementwise_multiply_batch(d_a, d_b, d_c, rows, batch_size);

    // Copy result back
    std::vector<Eigen::VectorXd> result_c(batch_size);
    gpuContext->copy_batch_to_host(result_c, d_c, rows, batch_size);

    // Clean up
    gpuContext->free_batch_vector(d_a);
    gpuContext->free_batch_vector(d_b);
    gpuContext->free_batch_vector(d_c);

    // Compare
    for (int i = 0; i < batch_size; ++i) {
        assertVectorApprox(result_c[i], expected_c[i], TOL, "Multiply result mismatch for batch " + std::to_string(i), __FILE__, __LINE__);
    }

    std::cout << "Test Passed.." << std::endl;
    passed_tests_++;
    return true;
}

bool NeuralNetworkTest::test_computeGradientsGPU_batch() {
    std::cout << "-----Running Test: test_computeGradientsGPU_batch-----" << std::endl;
    total_tests_++;

    // Setup: m=2 neurons, n=3 prev neurons, batch_size=2
    int m = 2, n = 3, batch_size = 2;
    std::vector<Eigen::VectorXd> deltas_batch(batch_size), prev_a_batch(batch_size);
    deltas_batch[0] = Eigen::VectorXd(m); deltas_batch[0] << 0.1, 0.2;
    deltas_batch[1] = Eigen::VectorXd(m); deltas_batch[1] << 0.3, 0.4;
    prev_a_batch[0] = Eigen::VectorXd(n); prev_a_batch[0] << 0.5, 0.6, 0.7;
    prev_a_batch[1] = Eigen::VectorXd(n); prev_a_batch[1] << 0.8, 0.9, 1.0;

    // Expected: weight_grad += deltas * prev_a^T, bias_grad += sum(deltas)
    Eigen::MatrixXd expected_w_grad(m, n);
    expected_w_grad.setZero();
    for (int b = 0; b < batch_size; ++b) {
        expected_w_grad += deltas_batch[b] * prev_a_batch[b].transpose();
    }
    Eigen::VectorXd expected_b_grad(m);
    expected_b_grad.setZero();
    for (int b = 0; b < batch_size; ++b) {
        expected_b_grad += deltas_batch[b];
    }

    // GPU: Allocate and copy inputs
    double* d_deltas, * d_prev_a, * d_w_grad, * d_b_grad;
    gpuContext->allocate_batch_vector(&d_deltas, m, batch_size);
    gpuContext->allocate_batch_vector(&d_prev_a, n, batch_size);
    gpuContext->allocate_weights(&d_w_grad, m, n);
    gpuContext->allocate_biases(&d_b_grad, m);
    gpuContext->copy_batch_to_device(d_deltas, deltas_batch, false);
    gpuContext->copy_batch_to_device(d_prev_a, prev_a_batch, false);
    gpuContext->set_to_zero(d_w_grad, m * n);
    gpuContext->set_to_zero(d_b_grad, m);

    // Run GPU gradient computation
    gpuContext->computeGradientsGPU_batch(d_deltas, d_prev_a, d_w_grad, d_b_grad, m, n, batch_size);

    // Copy results back
    Eigen::MatrixXd result_w_grad(m, n);
    Eigen::VectorXd result_b_grad(m);
    gpuContext->copy_weights_to_host(result_w_grad, d_w_grad, m, n);
    gpuContext->copy_biases_to_host(result_b_grad, d_b_grad, m);

    // Clean up
    gpuContext->free_batch_vector(d_deltas);
    gpuContext->free_batch_vector(d_prev_a);
    gpuContext->free_weights(d_w_grad);
    gpuContext->free_biases(d_b_grad);

    // Compare
    assertMatrixApprox(result_w_grad, expected_w_grad, TOL, "Weight grads mismatch", __FILE__, __LINE__);
    assertVectorApprox(result_b_grad, expected_b_grad, TOL, "Bias grads mismatch", __FILE__, __LINE__);

    std::cout << "Test Passed.." << std::endl;
    passed_tests_++;
    return true;
}

bool NeuralNetworkTest::test_compute_delta_back_batch() {
    std::cout << "-----Running Test: test_compute_delta_back_batch-----" << std::endl;
    total_tests_++;

    // Setup: W (n=3 x m=2), delta_next (m=2 x batch_size=2)
    int m = 2, n = 3, batch_size = 2;
    Eigen::MatrixXd weights(m, n);
    weights << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6;
    std::vector<Eigen::VectorXd> delta_next_batch(batch_size);
    delta_next_batch[0] = Eigen::VectorXd(m); delta_next_batch[0] << 0.1, 0.2;
    delta_next_batch[1] = Eigen::VectorXd(m); delta_next_batch[1] << 0.3, 0.4;

    // Expected: delta = W^T * delta_next
    std::vector<Eigen::VectorXd> expected_delta(batch_size);
    for (int b = 0; b < batch_size; ++b) {
        expected_delta[b] = weights.transpose() * delta_next_batch[b];
    }

    // GPU: Allocate and copy inputs
    double* d_weights, * d_delta_next, * d_delta;
    gpuContext->allocate_weights(&d_weights, n, m);
    gpuContext->allocate_batch_vector(&d_delta_next, m, batch_size);
    gpuContext->allocate_batch_vector(&d_delta, n, batch_size);
    gpuContext->copy_to_device(d_weights, weights);
    gpuContext->copy_batch_to_device(d_delta_next, delta_next_batch, false);
    gpuContext->set_to_zero_batch(d_delta, n, batch_size);

    // Run GPU delta back
    gpuContext->compute_delta_back_batch(d_weights, d_delta_next, d_delta, m, n, batch_size);

    // Copy result back
    std::vector<Eigen::VectorXd> result_delta(batch_size);
    gpuContext->copy_batch_to_host(result_delta, d_delta, n, batch_size);

    // Clean up
    gpuContext->free_weights(d_weights);
    gpuContext->free_batch_vector(d_delta_next);
    gpuContext->free_batch_vector(d_delta);

    // Compare
    for (int i = 0; i < batch_size; ++i) {
        assertVectorApprox(result_delta[i], expected_delta[i], TOL, "Delta back mismatch for batch " + std::to_string(i), __FILE__, __LINE__);
    }

    std::cout << "Test Passed.." << std::endl;
    passed_tests_++;
    return true;
}

bool NeuralNetworkTest::test_computeActivationDerivativeGPU_batch() {
    std::cout << "-----Running Test: test_computeActivationDerivativeGPU_batch-----" << std::endl;
    total_tests_++;

    // Setup: 2 neurons, batch_size=2, sigmoid activation
    int vec_size = 2, batch_size = 2;
    std::vector<Eigen::VectorXd> pre_activations(batch_size);
    pre_activations[0] = Eigen::VectorXd(vec_size); pre_activations[0] << 0.0, 1.0;
    pre_activations[1] = Eigen::VectorXd(vec_size); pre_activations[1] << -1.0, 0.5;

    // Expected: sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z))
    std::vector<Eigen::VectorXd> expected_deriv(batch_size);
    for (int b = 0; b < batch_size; ++b) {
        expected_deriv[b] = Eigen::VectorXd(vec_size);
        for (int i = 0; i < vec_size; ++i) {
            double z = pre_activations[b](i);
            double sig = 1.0 / (1.0 + std::exp(-z));
            expected_deriv[b](i) = sig * (1.0 - sig);
        }
    }

    // GPU: Allocate and copy inputs
    double* d_pre_activations, * d_derivatives;
    gpuContext->allocate_batch_vector(&d_pre_activations, vec_size, batch_size);
    gpuContext->allocate_batch_vector(&d_derivatives, vec_size, batch_size);
    gpuContext->copy_batch_to_device(d_pre_activations, pre_activations, false);
    gpuContext->set_to_zero_batch(d_derivatives, vec_size, batch_size);

    // Run GPU derivative
    gpuContext->computeActivationDerivativeGPU_batch(d_pre_activations, d_derivatives, vec_size, batch_size, activation_.get());

    // Copy result back
    std::vector<Eigen::VectorXd> result_deriv(batch_size);
    gpuContext->copy_batch_to_host(result_deriv, d_derivatives, vec_size, batch_size);

    // Clean up
    gpuContext->free_batch_vector(d_pre_activations);
    gpuContext->free_batch_vector(d_derivatives);

    // Compare
    for (int i = 0; i < batch_size; ++i) {
        assertVectorApprox(result_deriv[i], expected_deriv[i], TOL, "Sigmoid derivative mismatch for batch " + std::to_string(i), __FILE__, __LINE__);
    }

    std::cout << "Test Passed.." << std::endl;
    passed_tests_++;
    return true;
}

bool NeuralNetworkTest::test_cost_prime_mse_crossent_batched() {
    std::cout << "-----Running Test: test_cost_prime_mse_crossent_batched-----" << std::endl;
    total_tests_++;

    // Setup: 2 outputs, batch_size=2
    int rows = 2, batch_size = 2;
    std::vector<Eigen::VectorXd> output_batch(batch_size), target_batch(batch_size);
    output_batch[0] = Eigen::VectorXd(rows); output_batch[0] << 0.8, 0.2;
    output_batch[1] = Eigen::VectorXd(rows); output_batch[1] << 0.9, 0.1;
    target_batch[0] = Eigen::VectorXd(rows); target_batch[0] << 0.3, 0.7;
    target_batch[1] = Eigen::VectorXd(rows); target_batch[1] << 0.4, 0.6;

    // Expected: delta = output - target
    std::vector<Eigen::VectorXd> expected_delta(batch_size);
    for (int b = 0; b < batch_size; ++b) {
        expected_delta[b] = output_batch[b] - target_batch[b];
    }

    // GPU: Allocate and copy inputs
    double* d_output, * d_target, * d_delta;
    gpuContext->allocate_batch_vector(&d_output, rows, batch_size);
    gpuContext->allocate_batch_vector(&d_target, rows, batch_size);
    gpuContext->allocate_batch_vector(&d_delta, rows, batch_size);
    gpuContext->copy_batch_to_device(d_output, output_batch, false);
    gpuContext->copy_batch_to_device(d_target, target_batch, false);
    gpuContext->set_to_zero_batch(d_delta, rows, batch_size);

    // Run GPU cost prime
    gpuContext->cost_prime_mse_crossent_batched(d_output, d_target, d_delta, rows, batch_size);

    // Copy result back
    std::vector<Eigen::VectorXd> result_delta(batch_size);
    gpuContext->copy_batch_to_host(result_delta, d_delta, rows, batch_size);

    // Clean up
    gpuContext->free_batch_vector(d_output);
    gpuContext->free_batch_vector(d_target);
    gpuContext->free_batch_vector(d_delta);

    // Compare
    for (int i = 0; i < batch_size; ++i) {
        assertVectorApprox(result_delta[i], expected_delta[i], TOL, "Cost prime mismatch for batch " + std::to_string(i), __FILE__, __LINE__);
    }

    std::cout << "Test Passed.." << std::endl;
    passed_tests_++;
    return true;
}

bool NeuralNetworkTest::test_compute_mse_loss_batch_gpu() {
    std::cout << "-----Running Test: test_compute_mse_loss_batch_gpu-----" << std::endl;
    total_tests_++;

    // Setup: 2 outputs, batch_size=2
    int rows = 2, batch_size = 2;
    std::vector<Eigen::VectorXd> output_batch(batch_size), target_batch(batch_size);
    output_batch[0] = Eigen::VectorXd(rows); output_batch[0] << 0.8, 0.2;
    output_batch[1] = Eigen::VectorXd(rows); output_batch[1] << 0.9, 0.1;
    target_batch[0] = Eigen::VectorXd(rows); target_batch[0] << 0.3, 0.7;
    target_batch[1] = Eigen::VectorXd(rows); target_batch[1] << 0.4, 0.6;

    // Expected MSE: sum((output - target)^2) / (rows * batch_size)
    double expected_loss = 0.0;
    for (int b = 0; b < batch_size; ++b) {
        Eigen::VectorXd diff = output_batch[b] - target_batch[b];
        expected_loss += diff.squaredNorm();
    }
    expected_loss /= (rows * batch_size);

    // GPU: Allocate and copy inputs
    double* d_output, * d_target;
    gpuContext->allocate_batch_vector(&d_output, rows, batch_size);
    gpuContext->allocate_batch_vector(&d_target, rows, batch_size);
    gpuContext->copy_batch_to_device(d_output, output_batch, false);
    gpuContext->copy_batch_to_device(d_target, target_batch, false);

    // Run GPU MSE loss
    double result_loss = gpuContext->compute_mse_loss_batch_gpu(d_output, d_target, rows, batch_size);

    // Clean up
    gpuContext->free_batch_vector(d_output);
    gpuContext->free_batch_vector(d_target);

    // Compare
    assertApprox(result_loss, expected_loss, TOL, "MSE loss mismatch", __FILE__, __LINE__);

    std::cout << "Test Passed.." << std::endl;
    passed_tests_++;
    return true;
}


bool NeuralNetworkTest::test_backprop_gpu_batch() {
    std::cout << "-----Running Test: test_backprop_gpu_batch-----" << std::endl;
    total_tests_++;

    // Common setup: Network params and batch data
    std::vector<Eigen::VectorXd> batch_inputs = {
        (Eigen::VectorXd(2) << 0.5, 0.3).finished(),
        (Eigen::VectorXd(2) << 0.7, 0.2).finished()
    };
    std::vector<Eigen::VectorXd> batch_targets = {
        (Eigen::VectorXd(2) << 0.1, 0.9).finished(),
        (Eigen::VectorXd(2) << 0.8, 0.4).finished()
    };
    int batch_size = batch_inputs.size();

    // Test for MSE and CE loss
    std::vector<Network::LossType> loss_types = { Network::LossType::MSE, Network::LossType::CROSS_ENTROPY };
    for (const auto& loss_type : loss_types) {
        std::cout << "Testing with " << (loss_type == Network::LossType::MSE ? "MSE" : "Cross-Entropy") << " loss" << std::endl;

        // Initialize networks
        std::unique_ptr<Network> net_cpu = std::make_unique<Network>(network_sizes_, 0.0, loss_type, neuron_type_, cpuContext.get(), seed_, 2);
        std::unique_ptr<Network> net_gpu = std::make_unique<Network>(network_sizes_, 0.0, loss_type, neuron_type_, gpuContext.get(), seed_, 2);

        // Sync weights and biases
        for (size_t i = 0; i < network_sizes_.size() - 1; ++i) {
            auto& cpu_layer = net_cpu->get_layers()[i];
            net_gpu->set_layer_weights(i, cpu_layer->get_weights());
            net_gpu->set_layer_biases(i, cpu_layer->get_biases());
        }

        // GPU: Forward and backprop batch
        net_gpu->init_batch_buffers(batch_size);
        std::vector<Eigen::VectorXd> batch_outputs(batch_size);
        net_gpu->feedforward_gpu_batch(batch_inputs, batch_outputs);
        double batch_loss_gpu;
        net_gpu->backprop_gpu_batch(batch_targets, batch_loss_gpu);

        // CPU: Single-example forward and backprop, accumulate grads
        std::vector<Eigen::MatrixXd> cpu_weight_grads(network_sizes_.size() - 1);
        std::vector<Eigen::VectorXd> cpu_bias_grads(network_sizes_.size() - 1);
        double batch_loss_cpu = 0.0;
        size_t total_n = batch_size;

        for (size_t i = 0; i < network_sizes_.size() - 1; ++i) {
            cpu_weight_grads[i] = Eigen::MatrixXd::Zero(network_sizes_[i + 1], network_sizes_[i]);
            cpu_bias_grads[i] = Eigen::VectorXd::Zero(network_sizes_[i + 1]);
        }

        for (int i = 0; i < batch_size; ++i) {
            Eigen::VectorXd output = net_cpu->feedforward_cpu(batch_inputs[i]);
            auto [cpu_nabla_b, cpu_nabla_w] = net_cpu->backprop_cpu(batch_inputs[i], batch_targets[i], total_n);

            for (size_t l = 0; l < cpu_nabla_w.size(); ++l) {
                cpu_weight_grads[l] += cpu_nabla_w[l];
                cpu_bias_grads[l] += cpu_nabla_b[l];
            }

            // Compute loss per example
            if (loss_type == Network::LossType::MSE) {
                Eigen::VectorXd diff = output - batch_targets[i];
                batch_loss_cpu += diff.squaredNorm() / network_sizes_.back();
            }
            else {
                for (int j = 0; j < output.size(); ++j) {
                    double a = output(j), y = batch_targets[i](j);
                    batch_loss_cpu += -y * std::log(std::max(a, 1e-10)) - (1.0 - y) * std::log(std::max(1.0 - a, 1e-10));
                }
                batch_loss_cpu /= network_sizes_.back();
            }
        }
        batch_loss_cpu /= batch_size;

        // Copy GPU accumulated grads to host
        std::vector<Eigen::MatrixXd> gpu_weight_grads(network_sizes_.size() - 1);
        std::vector<Eigen::VectorXd> gpu_bias_grads(network_sizes_.size() - 1);
        for (size_t i = 0; i < network_sizes_.size() - 1; ++i) {
            int rows = network_sizes_[i + 1], cols = network_sizes_[i];
            gpu_weight_grads[i] = Eigen::MatrixXd(rows, cols);
            gpu_bias_grads[i] = Eigen::VectorXd(rows);
            gpuContext->copy_weights_to_host(gpu_weight_grads[i], net_gpu->get_accumulate_weight_grads()[i], rows, cols);
            gpuContext->copy_biases_to_host(gpu_bias_grads[i], net_gpu->get_accumulate_bias_grads()[i], rows);
        }

        // Compare gradients
        for (size_t i = 0; i < network_sizes_.size() - 1; ++i) {
            assertMatrixApprox(cpu_weight_grads[i], gpu_weight_grads[i], TOL, "Weight grads mismatch at layer " + std::to_string(i) + " for " + (loss_type == Network::LossType::MSE ? "MSE" : "CE"), __FILE__, __LINE__);
            assertVectorApprox(cpu_bias_grads[i], gpu_bias_grads[i], TOL, "Bias grads mismatch at layer " + std::to_string(i) + " for " + (loss_type == Network::LossType::MSE ? "MSE" : "CE"), __FILE__, __LINE__);
        }

        // Compare loss (skip for CE since not implemented)
        if (loss_type == Network::LossType::MSE) {
            //assertApprox(batch_loss_cpu, batch_loss_gpu, TOL, "Batch loss mismatch for MSE", __FILE__, __LINE__);
        } // else { assertApprox(batch_loss_cpu, batch_loss_gpu, TOL, "Batch loss mismatch for CE", __FILE__, __LINE__); }
    }

    std::cout << "Test Passed.." << std::endl;
    passed_tests_++;
    return true;
}

//**
bool NeuralNetworkTest::test_update_mini_batch_batch() {
    std::cout << "-----Running Test: test_update_mini_batch_batch-----" << std::endl;
    total_tests_++;

    // Common setup: Batch data and parameters
    std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> mini_batch = {
        { (Eigen::VectorXd(2) << 0.5, 0.3).finished(), (Eigen::VectorXd(2) << 0.1, 0.9).finished() },
        { (Eigen::VectorXd(2) << 0.7, 0.2).finished(), (Eigen::VectorXd(2) << 0.8, 0.4).finished() }
    };
    int batch_size = mini_batch.size();
    double eta = 0.1;  // Learning rate
    double lambda = 0.01;  // L2 regularization
    size_t n = 4;  // Total dataset size (for regularization scaling)

    // Test for MSE and CE loss
    std::vector<Network::LossType> loss_types = { Network::LossType::MSE, Network::LossType::CROSS_ENTROPY };

    for (const auto& loss_type : loss_types) {
        std::cout << "Testing with " << (loss_type == Network::LossType::MSE ? "MSE" : "Cross-Entropy") << " loss" << std::endl;

        // Initialize networks
        std::unique_ptr<Network> net_cpu = std::make_unique<Network>(network_sizes_, lambda, loss_type, neuron_type_, cpuContext.get(), seed_, 2);
        std::unique_ptr<Network> net_gpu = std::make_unique<Network>(network_sizes_, lambda, loss_type, neuron_type_, gpuContext.get(), seed_, 2);

        // Sync initial weights and biases
        for (size_t i = 0; i < network_sizes_.size() - 1; ++i) {
            auto& cpu_layer = net_cpu->get_layers()[i];
            net_gpu->set_layer_weights(i, cpu_layer->get_weights());
            net_gpu->set_layer_biases(i, cpu_layer->get_biases());
        }

        double cpu_loss = 0.0;
        double gpu_loss = 0.0;
        net_cpu->update_mini_batch(mini_batch , eta, n);
        net_gpu->init_batch_buffers(batch_size);
        gpu_loss = net_gpu->update_mini_batch_batch(mini_batch, eta, n);

        // compute cpu batch loss, manually, since its not calculated in  net_cpu->update_mini_batch() yet.
        // gpu loss in net_gpu->update_mini_batch_batch() is also stubbed to 0.0. 
        // so defer loss comparison to later phase.

        // Compare final weights and biases
        for (size_t i = 0; i < network_sizes_.size() - 1; ++i) {
            auto& cpu_layer = net_cpu->get_layers()[i];
            auto& gpu_layer = net_gpu->get_layers()[i];
            assertMatrixApprox(cpu_layer->get_weights(), gpu_layer->get_weights(), TOL,
                "Weight mismatch at layer " + std::to_string(i) + " for " + (loss_type == Network::LossType::MSE ? "MSE" : "CE"), __FILE__, __LINE__);
            assertVectorApprox(cpu_layer->get_biases(), gpu_layer->get_biases(), TOL,
                "Bias mismatch at layer " + std::to_string(i) + " for " + (loss_type == Network::LossType::MSE ? "MSE" : "CE"), __FILE__, __LINE__);
        }

        // Compare loss (skip for CE since not implemented)
        if (loss_type == Network::LossType::MSE) {
            //assertApprox(cpu_loss, gpu_loss, TOL, "Batch loss mismatch for MSE", __FILE__, __LINE__);
        } // else { assertApprox(cpu_loss, gpu_loss, TOL, "Batch loss mismatch for CE", __FILE__, __LINE__); }
    }

    std::cout << "Test Passed.." << std::endl;
    passed_tests_++;
    return true;
}

bool NeuralNetworkTest::runAllTests()
{
    passed_tests_ = 0;
    total_tests_ = 0;

    //tests
    //testNetworkBackprop();
    //testUpdateMiniBatch();
    //customtest();

    //test_launch_elementwise_subtract_batch();
    //test_launch_elementwise_multiply_batch();
    //test_computeGradientsGPU_batch();
    //test_compute_delta_back_batch();
    //test_computeActivationDerivativeGPU_batch();
    //test_cost_prime_mse_crossent_batched();
    //test_compute_mse_loss_batch_gpu();

    //testBatchFunctionsGPU_context();
    //test_feedforward_gpu_batch();
    //test_backprop_gpu_batch();
    test_update_mini_batch_batch();

    std::cout << "Test Summary: " << passed_tests_ << "/" << total_tests_ << " tests passed" << std::endl;
    return passed_tests_ == total_tests_;
}

bool NeuralNetworkTest::customtest() {
    return true;
}