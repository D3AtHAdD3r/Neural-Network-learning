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


bool NeuralNetworkTest::runAllTests()
{
    passed_tests_ = 0;
    total_tests_ = 0;
    
    //tests
    testNetworkBackprop();
    testUpdateMiniBatch();
    //customtest();

    std::cout << "Test Summary: " << passed_tests_ << "/" << total_tests_ << " tests passed" << std::endl;
    return passed_tests_ == total_tests_;
}



bool NeuralNetworkTest::customtest() {

    double lambda = 0.0;

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
    //double normcpu = net_cpu.update_mini_batch(mini_batch, eta, n);
    double normgpu = net_gpu.update_mini_batch(mini_batch, eta, n);


    //Debug
    {
        if (true) {
            const auto& layers = net_gpu.get_layers();
            int rows, cols;
            for (size_t i = 0; i < net_gpu.get_num_layers() - 1; ++i) {
                rows = net_gpu.get_layer_sizes()[i + 1];
                cols = net_gpu.get_layer_sizes()[i];
                gpuContext->debugPrint(layers[i]->get_d_weights(), rows * cols);
                displayMatrixXd(layers[i]->get_weights());
            }
        }

    }

    return true;
}





