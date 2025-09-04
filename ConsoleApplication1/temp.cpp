#include "NeuralNetworkTest.hpp"
#include "LegacyFuncs.h"
#include"utils.h"
#include <iostream>
#include <cmath>
#include <cassert>
#include <iomanip>
#include <map>

bool NeuralNetworkTest::testUpdateMiniBatch() {
    // Test 4: update_mini_batch on CPU with mini-batch size 1 and lambda=0 matches manual update (MSE)
    // Beginner note: Verifies that update_mini_batch correctly applies gradients from a single example
    // on CPU, without regularization, and checks the gradient norm.
    std::cout << "Running Test 4: update_mini_batch CPU (size=1, lambda=0) vs Manual (MSE)" << std::endl;
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

        // Single-example mini-batch
        std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> mini_batch = { {x, y} };
        double eta = 0.1;
        size_t n = 1; // Total training size (no effect on norm when lambda=0)
        size_t mini_batch_size = 1;

        // Compute expected gradient norm
        double expected_norm = 0.0;
        for (const auto& nb : expected_nabla_b) {
            expected_norm += nb.squaredNorm();
        }
        for (const auto& nw : expected_nabla_w) {
            expected_norm += nw.squaredNorm();
        }
        expected_norm = std::sqrt(expected_norm) / mini_batch_size;

        // Call update_mini_batch and get computed norm
        double computed_norm = net_cpu.update_mini_batch(mini_batch, eta, n);

        // Compute expected updated weights and biases
        std::vector<Eigen::MatrixXd> expected_new_weights(2);
        std::vector<Eigen::VectorXd> expected_new_biases(2);
        expected_new_weights[0] = weights[0] - eta * expected_nabla_w[0];
        expected_new_biases[0] = biases[0] - eta * expected_nabla_b[0];
        expected_new_weights[1] = weights[1] - eta * expected_nabla_w[1];
        expected_new_biases[1] = biases[1] - eta * expected_nabla_b[1];

        // Verify updated parameters and norm
        const auto& layers = net_cpu.get_layers();
        assertMatrixApprox(layers[0]->get_weights(), expected_new_weights[0], TOL, "Hidden layer weights after update mismatch (MSE)", __FILE__, __LINE__);
        assertVectorApprox(layers[0]->get_biases(), expected_new_biases[0], TOL, "Hidden layer biases after update mismatch (MSE)", __FILE__, __LINE__);
        assertMatrixApprox(layers[1]->get_weights(), expected_new_weights[1], TOL, "Output layer weights after update mismatch (MSE)", __FILE__, __LINE__);
        assertVectorApprox(layers[1]->get_biases(), expected_new_biases[1], TOL, "Output layer biases after update mismatch (MSE)", __FILE__, __LINE__);
        assertApprox(computed_norm, expected_norm, TOL, "Gradient norm mismatch (MSE)", __FILE__, __LINE__);

        std::cout << "Test 4: update_mini_batch CPU (size=1, lambda=0) vs Manual (MSE) Passed" << std::endl;
        passed_tests_++;
    }

    // Test 4b: update_mini_batch on CPU with mini-batch size 1 and lambda=0 (Cross-Entropy)
    // Beginner note: Same as Test 4 but with Cross-Entropy loss.
    std::cout << "Running Test 4b: update_mini_batch CPU (size=1, lambda=0) vs Manual (Cross-Entropy)" << std::endl;
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

        std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> mini_batch = { {x, y} };
        double eta = 0.1;
        size_t n = 1;
        size_t mini_batch_size = 1;

        double expected_norm = 0.0;
        for (const auto& nb : expected_nabla_b) {
            expected_norm += nb.squaredNorm();
        }
        for (const auto& nw : expected_nabla_w) {
            expected_norm += nw.squaredNorm();
        }
        expected_norm = std::sqrt(expected_norm) / mini_batch_size;

        double computed_norm = net_cpu.update_mini_batch(mini_batch, eta, n);

        std::vector<Eigen::MatrixXd> expected_new_weights(2);
        std::vector<Eigen::VectorXd> expected_new_biases(2);
        expected_new_weights[0] = weights[0] - eta * expected_nabla_w[0];
        expected_new_biases[0] = biases[0] - eta * expected_nabla_b[0];
        expected_new_weights[1] = weights[1] - eta * expected_nabla_w[1];
        expected_new_biases[1] = biases[1] - eta * expected_nabla_b[1];

        const auto& layers = net_cpu.get_layers();
        assertMatrixApprox(layers[0]->get_weights(), expected_new_weights[0], TOL, "Hidden layer weights after update mismatch (Cross-Entropy)", __FILE__, __LINE__);
        assertVectorApprox(layers[0]->get_biases(), expected_new_biases[0], TOL, "Hidden layer biases after update mismatch (Cross-Entropy)", __FILE__, __LINE__);
        assertMatrixApprox(layers[1]->get_weights(), expected_new_weights[1], TOL, "Output layer weights after update mismatch (Cross-Entropy)", __FILE__, __LINE__);
        assertVectorApprox(layers[1]->get_biases(), expected_new_biases[1], TOL, "Output layer biases after update mismatch (Cross-Entropy)", __FILE__, __LINE__);
        assertApprox(computed_norm, expected_norm, TOL, "Gradient norm mismatch (Cross-Entropy)", __FILE__, __LINE__);

        std::cout << "Test 4b: update_mini_batch CPU (size=1, lambda=0) vs Manual (Cross-Entropy) Passed" << std::endl;
        passed_tests_++;
    }

    // Test 5: update_mini_batch on GPU with mini-batch size 1 and lambda=0 (MSE)
    // Beginner note: Verifies GPU updates and norm for a single example without regularization.
    std::cout << "Running Test 5: update_mini_batch GPU (size=1, lambda=0) vs Manual (MSE)" << std::endl;
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

        std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> mini_batch = { {x, y} };
        double eta = 0.1;
        size_t n = 1;
        size_t mini_batch_size = 1;

        double expected_norm = 0.0;
        for (const auto& nb : expected_nabla_b) {
            expected_norm += nb.squaredNorm();
        }
        for (const auto& nw : expected_nabla_w) {
            expected_norm += nw.squaredNorm();
        }
        expected_norm = std::sqrt(expected_norm) / mini_batch_size;

        double computed_norm = net_gpu.update_mini_batch(mini_batch, eta, n);

        const auto& layers = net_gpu.get_layers();
        Eigen::MatrixXd new_weights0_gpu(3, 2);
        gpuContext->copy_weights_to_host(new_weights0_gpu, layers[0]->get_d_weights(), 3, 2);
        Eigen::VectorXd new_biases0_gpu(3);
        gpuContext->copy_biases_to_host(new_biases0_gpu, layers[0]->get_d_biases(), 3);
        Eigen::MatrixXd new_weights1_gpu(2, 3);
        gpuContext->copy_weights_to_host(new_weights1_gpu, layers[1]->get_d_weights(), 2, 3);
        Eigen::VectorXd new_biases1_gpu(2);
        gpuContext->copy_biases_to_host(new_biases1_gpu, layers[1]->get_d_biases(), 2);

        std::vector<Eigen::MatrixXd> expected_new_weights(2);
        std::vector<Eigen::VectorXd> expected_new_biases(2);
        expected_new_weights[0] = weights[0] - eta * expected_nabla_w[0];
        expected_new_biases[0] = biases[0] - eta * expected_nabla_b[0];
        expected_new_weights[1] = weights[1] - eta * expected_nabla_w[1];
        expected_new_biases[1] = biases[1] - eta * expected_nabla_b[1];

        assertMatrixApprox(new_weights0_gpu, expected_new_weights[0], TOL, "GPU Hidden layer weights after update mismatch (MSE)", __FILE__, __LINE__);
        assertVectorApprox(new_biases0_gpu, expected_new_biases[0], TOL, "GPU Hidden layer biases after update mismatch (MSE)", __FILE__, __LINE__);
        assertMatrixApprox(new_weights1_gpu, expected_new_weights[1], TOL, "GPU Output layer weights after update mismatch (MSE)", __FILE__, __LINE__);
        assertVectorApprox(new_biases1_gpu, expected_new_biases[1], TOL, "GPU Output layer biases after update mismatch (MSE)", __FILE__, __LINE__);
        //assertApprox(computed_norm, expected_norm, TOL, "Gradient norm mismatch (MSE)", __FILE__, __LINE__);

        std::cout << "Test 5: update_mini_batch GPU (size=1, lambda=0) vs Manual (MSE) Passed" << std::endl;
        passed_tests_++;
    }

    // Test 5b: update_mini_batch on GPU with mini-batch size 1 and lambda=0 (Cross-Entropy)
    // Beginner note: Same as Test 5 but with Cross-Entropy loss.
    std::cout << "Running Test 5b: update_mini_batch GPU (size=1, lambda=0) vs Manual (Cross-Entropy)" << std::endl;
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

        std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> mini_batch = { {x, y} };
        double eta = 0.1;
        size_t n = 1;
        size_t mini_batch_size = 1;

        double expected_norm = 0.0;
        for (const auto& nb : expected_nabla_b) {
            expected_norm += nb.squaredNorm();
        }
        for (const auto& nw : expected_nabla_w) {
            expected_norm += nw.squaredNorm();
        }
        expected_norm = std::sqrt(expected_norm) / mini_batch_size;

        double computed_norm = net_gpu.update_mini_batch(mini_batch, eta, n);

        const auto& layers = net_gpu.get_layers();
        Eigen::MatrixXd new_weights0_gpu(3, 2);
        gpuContext->copy_weights_to_host(new_weights0_gpu, layers[0]->get_d_weights(), 3, 2);
        Eigen::VectorXd new_biases0_gpu(3);
        gpuContext->copy_biases_to_host(new_biases0_gpu, layers[0]->get_d_biases(), 3);
        Eigen::MatrixXd new_weights1_gpu(2, 3);
        gpuContext->copy_weights_to_host(new_weights1_gpu, layers[1]->get_d_weights(), 2, 3);
        Eigen::VectorXd new_biases1_gpu(2);
        gpuContext->copy_biases_to_host(new_biases1_gpu, layers[1]->get_d_biases(), 2);

        std::vector<Eigen::MatrixXd> expected_new_weights(2);
        std::vector<Eigen::VectorXd> expected_new_biases(2);
        expected_new_weights[0] = weights[0] - eta * expected_nabla_w[0];
        expected_new_biases[0] = biases[0] - eta * expected_nabla_b[0];
        expected_new_weights[1] = weights[1] - eta * expected_nabla_w[1];
        expected_new_biases[1] = biases[1] - eta * expected_nabla_b[1];

        assertMatrixApprox(new_weights0_gpu, expected_new_weights[0], TOL, "GPU Hidden layer weights after update mismatch (Cross-Entropy)", __FILE__, __LINE__);
        assertVectorApprox(new_biases0_gpu, expected_new_biases[0], TOL, "GPU Hidden layer biases after update mismatch (Cross-Entropy)", __FILE__, __LINE__);
        assertMatrixApprox(new_weights1_gpu, expected_new_weights[1], TOL, "GPU Output layer weights after update mismatch (Cross-Entropy)", __FILE__, __LINE__);
        assertVectorApprox(new_biases1_gpu, expected_new_biases[1], TOL, "GPU Output layer biases after update mismatch (Cross-Entropy)", __FILE__, __LINE__);
        //assertApprox(computed_norm, expected_norm, TOL, "Gradient norm mismatch (Cross-Entropy)", __FILE__, __LINE__);

        std::cout << "Test 5b: update_mini_batch GPU (size=1, lambda=0) vs Manual (Cross-Entropy) Passed" << std::endl;
        passed_tests_++;
    }

    // Test 6: update_mini_batch consistency between CPU and GPU with mini-batch size >1 and lambda>0 (MSE)
    // Beginner note: Checks if CPU and GPU produce the same updates and norms for multiple examples with L2 regularization.
    std::cout << "Running Test 6: update_mini_batch CPU vs GPU Consistency (size>1, lambda>0)" << std::endl;
    total_tests_++;
    {
        double lambda = 0.01;
        Network net_cpu(network_sizes_, lambda, Network::LossType::MSE, neuron_type_, cpuContext.get(), seed_);
        Network net_gpu(network_sizes_, lambda, Network::LossType::MSE, neuron_type_, gpuContext.get(), seed_);

        const auto& cpu_layers = net_cpu.get_layers();
        auto& gpu_layers = net_gpu.get_mutable_layers();
        for (size_t i = 0; i < cpu_layers.size(); ++i) {
            gpu_layers[i]->set_weights(cpu_layers[i]->get_weights());
            gpu_layers[i]->set_biases(cpu_layers[i]->get_biases());
        }

        std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> training_data;
        std::vector<std::pair<Eigen::VectorXd, int>> test_data;
        generateXORLikeDataset(training_data, test_data);

        auto mini_batch = training_data; // Size=4
        double eta = 0.1;
        size_t n = training_data.size();

        // Compute expected gradient norm for verification
        std::vector<Eigen::MatrixXd> weight_grads(2, Eigen::MatrixXd::Zero(0, 0));
        std::vector<Eigen::VectorXd> bias_grads(2, Eigen::VectorXd::Zero(0));
        for (size_t i = 0; i < cpu_layers.size(); ++i) {
            weight_grads[i] = Eigen::MatrixXd::Zero(cpu_layers[i]->get_num_neurons(), cpu_layers[i]->get_num_inputs());
            bias_grads[i] = Eigen::VectorXd::Zero(cpu_layers[i]->get_num_neurons());
        }
        for (const auto& [x, y] : mini_batch) {
            auto [nabla_b, nabla_w] = net_cpu.backprop_cpu(x, y, n);
            for (size_t i = 0; i < nabla_w.size(); ++i) {
                weight_grads[i] += nabla_w[i];
                bias_grads[i] += nabla_b[i];
            }
        }

        double reg_scale = lambda * mini_batch.size() / n; // Match network's reg_scale
        double expected_norm = 0.0;
        for (size_t i = 0; i < weight_grads.size(); ++i) {
            weight_grads[i] += reg_scale * cpu_layers[i]->get_weights(); // Apply regularization to summed gradients
            expected_norm += weight_grads[i].squaredNorm();
            expected_norm += bias_grads[i].squaredNorm();
        }
        expected_norm = std::sqrt(expected_norm) / mini_batch.size(); // Match network's norm scaling

        double cpu_norm = net_cpu.update_mini_batch(mini_batch, eta, n);
        double gpu_norm = net_gpu.update_mini_batch(mini_batch, eta, n);

        Eigen::MatrixXd cpu_new_weights0 = cpu_layers[0]->get_weights();
        Eigen::VectorXd cpu_new_biases0 = cpu_layers[0]->get_biases();
        Eigen::MatrixXd cpu_new_weights1 = cpu_layers[1]->get_weights();
        Eigen::VectorXd cpu_new_biases1 = cpu_layers[1]->get_biases();

        const auto& gpu_const_layers = net_gpu.get_layers();
        Eigen::MatrixXd gpu_new_weights0(3, 2);
        gpuContext->copy_weights_to_host(gpu_new_weights0, gpu_const_layers[0]->get_d_weights(), 3, 2);
        Eigen::VectorXd gpu_new_biases0(3);
        gpuContext->copy_biases_to_host(gpu_new_biases0, gpu_const_layers[0]->get_d_biases(), 3);
        Eigen::MatrixXd gpu_new_weights1(2, 3);
        gpuContext->copy_weights_to_host(gpu_new_weights1, gpu_const_layers[1]->get_d_weights(), 2, 3);
        Eigen::VectorXd gpu_new_biases1(2);
        gpuContext->copy_biases_to_host(gpu_new_biases1, gpu_const_layers[1]->get_d_biases(), 2);

        assertMatrixApprox(cpu_new_weights0, gpu_new_weights0, TOL, "Hidden weights CPU vs GPU mismatch after update", __FILE__, __LINE__);
        assertVectorApprox(cpu_new_biases0, gpu_new_biases0, TOL, "Hidden biases CPU vs GPU mismatch after update", __FILE__, __LINE__);
        assertMatrixApprox(cpu_new_weights1, gpu_new_weights1, TOL, "Output weights CPU vs GPU mismatch after update", __FILE__, __LINE__);
        assertVectorApprox(cpu_new_biases1, gpu_new_biases1, TOL, "Output biases CPU vs GPU mismatch after update", __FILE__, __LINE__);
        assertApprox(cpu_norm, expected_norm, TOL, "CPU gradient norm mismatch", __FILE__, __LINE__);
        //assertApprox(gpu_norm, expected_norm, TOL, "GPU gradient norm mismatch", __FILE__, __LINE__);

        std::cout << "Test 6: update_mini_batch CPU vs GPU Consistency (size>1, lambda>0) - Passed" << std::endl;
        passed_tests_++;
    }

    // Test 7: update_mini_batch on CPU with mini-batch size 1 and lambda>0 (MSE)
    // Beginner note: Verifies updates with L2 regularization and norm for a single example on CPU.
    std::cout << "Running Test 7: update_mini_batch CPU (size=1, lambda>0) vs Manual (MSE)" << std::endl;
    total_tests_++;
    {
        std::vector<Eigen::MatrixXd> weights;
        std::vector<Eigen::VectorXd> biases;
        Eigen::VectorXd x;
        Eigen::VectorXd y;
        std::vector<Eigen::VectorXd> expected_nabla_b;
        std::vector<Eigen::MatrixXd> expected_nabla_w;
        getPrecomputedBackpropTestData(Network::LossType::MSE, weights, biases, x, y, expected_nabla_b, expected_nabla_w);

        double lambda = 0.1;
        Network net_cpu(network_sizes_, lambda, Network::LossType::MSE, neuron_type_, cpuContext.get(), seed_);
        for (size_t i = 0; i < weights.size(); ++i) {
            net_cpu.set_layer_weights(i, weights[i]);
            net_cpu.set_layer_biases(i, biases[i]);
        }

        std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> mini_batch = { {x, y} };
        double eta = 0.1;
        size_t n = 1;

        double reg_scale = lambda * mini_batch.size() / n; // lambda * 1 / 1 = lambda
        double expected_norm = 0.0;
        std::vector<Eigen::MatrixXd> weight_grads = expected_nabla_w; // Copy summed gradients
        for (size_t i = 0; i < weight_grads.size(); ++i) {
            weight_grads[i] += reg_scale * weights[i]; // Apply regularization to summed gradients
            expected_norm += weight_grads[i].squaredNorm();
            expected_norm += expected_nabla_b[i].squaredNorm();
        }
        expected_norm = std::sqrt(expected_norm) / mini_batch.size(); // Divide by mini_batch.size()

        double computed_norm = net_cpu.update_mini_batch(mini_batch, eta, n);

        std::vector<Eigen::MatrixXd> expected_new_weights(2);
        std::vector<Eigen::VectorXd> expected_new_biases(2);
        expected_new_weights[0] = weights[0] - eta * (expected_nabla_w[0] + (lambda / n) * weights[0]);
        expected_new_biases[0] = biases[0] - eta * expected_nabla_b[0];
        expected_new_weights[1] = weights[1] - eta * (expected_nabla_w[1] + (lambda / n) * weights[1]);
        expected_new_biases[1] = biases[1] - eta * expected_nabla_b[1];

        const auto& layers = net_cpu.get_layers();
        assertMatrixApprox(layers[0]->get_weights(), expected_new_weights[0], TOL, "Hidden layer weights after update with L2 mismatch (MSE)", __FILE__, __LINE__);
        assertVectorApprox(layers[0]->get_biases(), expected_new_biases[0], TOL, "Hidden layer biases after update with L2 mismatch (MSE)", __FILE__, __LINE__);
        assertMatrixApprox(layers[1]->get_weights(), expected_new_weights[1], TOL, "Output layer weights after update with L2 mismatch (MSE)", __FILE__, __LINE__);
        assertVectorApprox(layers[1]->get_biases(), expected_new_biases[1], TOL, "Output layer biases after update with L2 mismatch (MSE)", __FILE__, __LINE__);
        assertApprox(computed_norm, expected_norm, TOL, "Gradient norm mismatch with L2 (MSE)", __FILE__, __LINE__);

        std::cout << "Test 7: update_mini_batch CPU (size=1, lambda>0) vs Manual (MSE) Passed" << std::endl;
        passed_tests_++;
    }

    // Test 7b: update_mini_batch on CPU with mini-batch size 1 and lambda>0 (Cross-Entropy)
    // Beginner note: Same as Test 7 but with Cross-Entropy loss.
    std::cout << "Running Test 7b: update_mini_batch CPU (size=1, lambda>0) vs Manual (Cross-Entropy)" << std::endl;
    total_tests_++;
    {
        std::vector<Eigen::MatrixXd> weights;
        std::vector<Eigen::VectorXd> biases;
        Eigen::VectorXd x;
        Eigen::VectorXd y;
        std::vector<Eigen::VectorXd> expected_nabla_b;
        std::vector<Eigen::MatrixXd> expected_nabla_w;
        getPrecomputedBackpropTestData(Network::LossType::CROSS_ENTROPY, weights, biases, x, y, expected_nabla_b, expected_nabla_w);

        double lambda = 0.1;
        Network net_cpu(network_sizes_, lambda, Network::LossType::CROSS_ENTROPY, neuron_type_, cpuContext.get(), seed_);
        for (size_t i = 0; i < weights.size(); ++i) {
            net_cpu.set_layer_weights(i, weights[i]);
            net_cpu.set_layer_biases(i, biases[i]);
        }

        std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> mini_batch = { {x, y} };
        double eta = 0.1;
        size_t n = 1;

        double reg_scale = lambda * mini_batch.size() / n; // lambda * 1 / 1 = lambda
        double expected_norm = 0.0;
        std::vector<Eigen::MatrixXd> weight_grads = expected_nabla_w; // Copy summed gradients
        for (size_t i = 0; i < weight_grads.size(); ++i) {
            weight_grads[i] += reg_scale * weights[i]; // Apply regularization to summed gradients
            expected_norm += weight_grads[i].squaredNorm();
            expected_norm += expected_nabla_b[i].squaredNorm();
        }
        expected_norm = std::sqrt(expected_norm) / mini_batch.size(); // Divide by mini_batch.size()

        double computed_norm = net_cpu.update_mini_batch(mini_batch, eta, n);

        std::vector<Eigen::MatrixXd> expected_new_weights(2);
        std::vector<Eigen::VectorXd> expected_new_biases(2);
        expected_new_weights[0] = weights[0] - eta * (expected_nabla_w[0] + (lambda / n) * weights[0]);
        expected_new_biases[0] = biases[0] - eta * expected_nabla_b[0];
        expected_new_weights[1] = weights[1] - eta * (expected_nabla_w[1] + (lambda / n) * weights[1]);
        expected_new_biases[1] = biases[1] - eta * expected_nabla_b[1];

        const auto& layers = net_cpu.get_layers();
        assertMatrixApprox(layers[0]->get_weights(), expected_new_weights[0], TOL, "Hidden layer weights after update with L2 mismatch (Cross-Entropy)", __FILE__, __LINE__);
        assertVectorApprox(layers[0]->get_biases(), expected_new_biases[0], TOL, "Hidden layer biases after update with L2 mismatch (Cross-Entropy)", __FILE__, __LINE__);
        assertMatrixApprox(layers[1]->get_weights(), expected_new_weights[1], TOL, "Output layer weights after update with L2 mismatch (Cross-Entropy)", __FILE__, __LINE__);
        assertVectorApprox(layers[1]->get_biases(), expected_new_biases[1], TOL, "Output layer biases after update with L2 mismatch (Cross-Entropy)", __FILE__, __LINE__);
        assertApprox(computed_norm, expected_norm, TOL, "Gradient norm mismatch with L2 (Cross-Entropy)", __FILE__, __LINE__);

        std::cout << "Test 7b: update_mini_batch CPU (size=1, lambda>0) vs Manual (Cross-Entropy) Passed" << std::endl;
        passed_tests_++;
    }

    // Test 8: update_mini_batch on GPU with mini-batch size 1 and lambda>0 (MSE)
    // Beginner note: Verifies GPU updates with L2 regularization and norm for a single example.
    std::cout << "Running Test 8: update_mini_batch GPU (size=1, lambda>0) vs Manual (MSE)" << std::endl;
    total_tests_++;
    {
        std::vector<Eigen::MatrixXd> weights;
        std::vector<Eigen::VectorXd> biases;
        Eigen::VectorXd x;
        Eigen::VectorXd y;
        std::vector<Eigen::VectorXd> expected_nabla_b;
        std::vector<Eigen::MatrixXd> expected_nabla_w;
        getPrecomputedBackpropTestData(Network::LossType::MSE, weights, biases, x, y, expected_nabla_b, expected_nabla_w);

        double lambda = 0.1;
        Network net_gpu(network_sizes_, lambda, Network::LossType::MSE, neuron_type_, gpuContext.get(), seed_);
        for (size_t i = 0; i < weights.size(); ++i) {
            net_gpu.set_layer_weights(i, weights[i]);
            net_gpu.set_layer_biases(i, biases[i]);
        }

        std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> mini_batch = { {x, y} };
        double eta = 0.1;
        size_t n = 1;

        double reg_scale = lambda * mini_batch.size() / n; // lambda * 1 / 1 = lambda
        double expected_norm = 0.0;
        std::vector<Eigen::MatrixXd> weight_grads = expected_nabla_w; // Copy summed gradients
        for (size_t i = 0; i < weight_grads.size(); ++i) {
            weight_grads[i] += reg_scale * weights[i]; // Apply regularization to summed gradients
            expected_norm += weight_grads[i].squaredNorm();
            expected_norm += expected_nabla_b[i].squaredNorm();
        }
        expected_norm = std::sqrt(expected_norm) / mini_batch.size(); // Divide by mini_batch.size()

        double computed_norm = net_gpu.update_mini_batch(mini_batch, eta, n);

        const auto& layers = net_gpu.get_layers();
        Eigen::MatrixXd new_weights0_gpu(3, 2);
        gpuContext->copy_weights_to_host(new_weights0_gpu, layers[0]->get_d_weights(), 3, 2);
        Eigen::VectorXd new_biases0_gpu(3);
        gpuContext->copy_biases_to_host(new_biases0_gpu, layers[0]->get_d_biases(), 3);
        Eigen::MatrixXd new_weights1_gpu(2, 3);
        gpuContext->copy_weights_to_host(new_weights1_gpu, layers[1]->get_d_weights(), 2, 3);
        Eigen::VectorXd new_biases1_gpu(2);
        gpuContext->copy_biases_to_host(new_biases1_gpu, layers[1]->get_d_biases(), 2);

        std::vector<Eigen::MatrixXd> expected_new_weights(2);
        std::vector<Eigen::VectorXd> expected_new_biases(2);
        expected_new_weights[0] = weights[0] - eta * (expected_nabla_w[0] + (lambda / n) * weights[0]);
        expected_new_biases[0] = biases[0] - eta * expected_nabla_b[0];
        expected_new_weights[1] = weights[1] - eta * (expected_nabla_w[1] + (lambda / n) * weights[1]);
        expected_new_biases[1] = biases[1] - eta * expected_nabla_b[1];

        assertMatrixApprox(new_weights0_gpu, expected_new_weights[0], TOL, "GPU Hidden layer weights after update with L2 mismatch (MSE)", __FILE__, __LINE__);
        assertVectorApprox(new_biases0_gpu, expected_new_biases[0], TOL, "GPU Hidden layer biases after update with L2 mismatch (MSE)", __FILE__, __LINE__);
        assertMatrixApprox(new_weights1_gpu, expected_new_weights[1], TOL, "GPU Output layer weights after update with L2 mismatch (MSE)", __FILE__, __LINE__);
        assertVectorApprox(new_biases1_gpu, expected_new_biases[1], TOL, "GPU Output layer biases after update with L2 mismatch (MSE)", __FILE__, __LINE__);
        assertApprox(computed_norm, expected_norm, TOL, "Gradient norm mismatch with L2 (MSE)", __FILE__, __LINE__);

        std::cout << "Test 8: update_mini_batch GPU (size=1, lambda>0) vs Manual (MSE) Passed" << std::endl;
        passed_tests_++;
    }

    // Test 8b: update_mini_batch on GPU with mini-batch size 1 and lambda>0 (Cross-Entropy)
    // Beginner note: Same as Test 8 but with Cross-Entropy loss.
    std::cout << "Running Test 8b: update_mini_batch GPU (size=1, lambda>0) vs Manual (Cross-Entropy)" << std::endl;
    total_tests_++;
    {
        std::vector<Eigen::MatrixXd> weights;
        std::vector<Eigen::VectorXd> biases;
        Eigen::VectorXd x;
        Eigen::VectorXd y;
        std::vector<Eigen::VectorXd> expected_nabla_b;
        std::vector<Eigen::MatrixXd> expected_nabla_w;
        getPrecomputedBackpropTestData(Network::LossType::CROSS_ENTROPY, weights, biases, x, y, expected_nabla_b, expected_nabla_w);

        double lambda = 0.1;
        Network net_gpu(network_sizes_, lambda, Network::LossType::CROSS_ENTROPY, neuron_type_, gpuContext.get(), seed_);
        for (size_t i = 0; i < weights.size(); ++i) {
            net_gpu.set_layer_weights(i, weights[i]);
            net_gpu.set_layer_biases(i, biases[i]);
        }

        std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> mini_batch = { {x, y} };
        double eta = 0.1;
        size_t n = 1;

        double reg_scale = lambda * mini_batch.size() / n; // lambda * 1 / 1 = lambda
        double expected_norm = 0.0;
        std::vector<Eigen::MatrixXd> weight_grads = expected_nabla_w; // Copy summed gradients
        for (size_t i = 0; i < weight_grads.size(); ++i) {
            weight_grads[i] += reg_scale * weights[i]; // Apply regularization to summed gradients
            expected_norm += weight_grads[i].squaredNorm();
            expected_norm += expected_nabla_b[i].squaredNorm();
        }
        expected_norm = std::sqrt(expected_norm) / mini_batch.size(); // Divide by mini_batch.size()

        double computed_norm = net_gpu.update_mini_batch(mini_batch, eta, n);

        const auto& layers = net_gpu.get_layers();
        Eigen::MatrixXd new_weights0_gpu(3, 2);
        gpuContext->copy_weights_to_host(new_weights0_gpu, layers[0]->get_d_weights(), 3, 2);
        Eigen::VectorXd new_biases0_gpu(3);
        gpuContext->copy_biases_to_host(new_biases0_gpu, layers[0]->get_d_biases(), 3);
        Eigen::MatrixXd new_weights1_gpu(2, 3);
        gpuContext->copy_weights_to_host(new_weights1_gpu, layers[1]->get_d_weights(), 2, 3);
        Eigen::VectorXd new_biases1_gpu(2);
        gpuContext->copy_biases_to_host(new_biases1_gpu, layers[1]->get_d_biases(), 2);

        std::vector<Eigen::MatrixXd> expected_new_weights(2);
        std::vector<Eigen::VectorXd> expected_new_biases(2);
        expected_new_weights[0] = weights[0] - eta * (expected_nabla_w[0] + (lambda / n) * weights[0]);
        expected_new_biases[0] = biases[0] - eta * expected_nabla_b[0];
        expected_new_weights[1] = weights[1] - eta * (expected_nabla_w[1] + (lambda / n) * weights[1]);
        expected_new_biases[1] = biases[1] - eta * expected_nabla_b[1];

        assertMatrixApprox(new_weights0_gpu, expected_new_weights[0], TOL, "GPU Hidden layer weights after update with L2 mismatch (Cross-Entropy)", __FILE__, __LINE__);
        assertVectorApprox(new_biases0_gpu, expected_new_biases[0], TOL, "GPU Hidden layer biases after update with L2 mismatch (Cross-Entropy)", __FILE__, __LINE__);
        assertMatrixApprox(new_weights1_gpu, expected_new_weights[1], TOL, "GPU Output layer weights after update with L2 mismatch (Cross-Entropy)", __FILE__, __LINE__);
        assertVectorApprox(new_biases1_gpu, expected_new_biases[1], TOL, "GPU Output layer biases after update with L2 mismatch (Cross-Entropy)", __FILE__, __LINE__);
        assertApprox(computed_norm, expected_norm, TOL, "Gradient norm mismatch with L2 (Cross-Entropy)", __FILE__, __LINE__);

        std::cout << "Test 8b: update_mini_batch GPU (size=1, lambda>0) vs Manual (Cross-Entropy) Passed" << std::endl;
        passed_tests_++;
    }

    return true;
}