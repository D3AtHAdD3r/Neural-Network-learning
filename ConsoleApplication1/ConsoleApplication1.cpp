#include "NeuralNetworkTest.hpp"
#include "Network.hpp"
#include "mnistLoader.h"
#include "utils.h"
//#include"CPUComputationContext.hpp"
//#include"GPUComputationContext.hpp"
#include <iostream>
#include <chrono> // Add this header


/*Changes made :
1. Added detailed metrics.
2. Implemented L2 Regularization and L2 Scaling.
3. Added a brief unit test for gradient checking
4. Added cross-entropy.
5. Added a brief activation interface with sigmoid only support currently.
*/

/*
Notes- 
Lambda Value:
lambda=0.01 may be too high for a small training set (3,000 MNIST samples). Common values for MNIST are 0.0001 to 0.001, scaled by the dataset size.
Reduce the learning rate over time (e.g., using a scheduler or smaller initial value like 0.1) to fine-tune the final accuracy beyond 90%.
Experiment with a slightly higher $\lambda$ (e.g., 0.001) to see if it helps prevent overfitting on the test set.
*/


int main() {
    // Example: [784, 30, 10] network
    std::vector<int> sizes = { 784, 30, 10 };
    std::string train_images = "data/train-images-idx3-ubyte";
    std::string train_labels = "data/train-labels-idx1-ubyte";
    std::string test_images = "data/t10k-images-idx3-ubyte";
    std::string test_labels = "data/t10k-labels-idx1-ubyte";

    // Load smaller dataset for testing
    auto training_data = load_mnist_training(train_images, train_labels, 15000);
    auto test_data = load_mnist_test(test_images, test_labels, 3000);

    // Create CPU and GPU computation contexts
    CPUComputationContext cpuContext;
    GPUComputationContext gpuContext;

    // Create networks with CPU and GPU contexts
    Network netCPU(sizes, 0.001, Network::LossType::CROSS_ENTROPY, Network::NeuronType::SIGMOID, &cpuContext);
    Network netGPU(sizes, 0.001, Network::LossType::CROSS_ENTROPY, Network::NeuronType::SIGMOID, &gpuContext);

    int epochs = 7;
    int mini_batch_size = 32;
    double eta = 1.5;

    // Train both network-
    // Train with CPU context and time it
    std::cout << "Training with Cpu context...\n";
    auto cpu_start = std::chrono::high_resolution_clock::now();
    netCPU.SGD(training_data, epochs, mini_batch_size, eta, &test_data, true);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_duration = cpu_end - cpu_start;
    std::cout << "CPU training completed in " << cpu_duration.count() << " seconds.\n";

    // Train with GPU context and time it
    std::cout << "Training with Gpu context...\n";
    auto gpu_start = std::chrono::high_resolution_clock::now();
    netGPU.SGD(training_data, epochs, mini_batch_size, eta, &test_data, true);
    auto gpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_duration = gpu_end - gpu_start;
    std::cout << "GPU training completed in " << gpu_duration.count() << " seconds.\n";

    return 0;
}


int main_9867fg() {
    // Use fixed network size for XOR-like dataset
    std::vector<int> xor_sizes = { 2, 3, 2 }; // Input: 2, Hidden: 3, Output: 2
    GPUComputationContext gpuContext;
    Network net(xor_sizes, 0.001, Network::LossType::CROSS_ENTROPY, Network::NeuronType::SIGMOID, &gpuContext);

    // Generate XOR-like dataset
    std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> training_data;
    std::vector<std::pair<Eigen::VectorXd, int>> test_data;
    NeuralNetworkTest tester;
    tester.generateXORLikeDataset(training_data, test_data);

    net.SGD(training_data, 1000, 1, 1.0, &test_data, true);

    return 0;
}



int main_23232() {

    // Default parameters
    NeuralNetworkTest tester;
    bool all_passed = tester.runAllTests();

    // Example with custom parameters
    /*std::vector<int> custom_sizes = { 2, 3, 2 };
    NeuralNetworkTest custom_tester(3, 4, 123, custom_sizes);

    bool all_passed;
    all_passed = custom_tester.runAllTests();

    if (all_passed) {
        std::cout << "All test suites passed!" << std::endl;
    }
    else {
        std::cerr << "Some tests failed." << std::endl;
        return 1;
    }*/

    return 0;
}
