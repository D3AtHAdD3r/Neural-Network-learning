#include "NeuralNetworkTest.hpp"
#include "Network.hpp"
#include "mnistLoader.h"
#include "utils.h"
#include <iostream>
#include <chrono> 
#include <vector>
#include <string>



/*
Notes- 
Lambda Value:
lambda=0.01 may be too high for a small training set (3,000 MNIST samples). Common values for MNIST are 0.0001 to 0.001, scaled by the dataset size.
Reduce the learning rate over time (e.g., using a scheduler or smaller initial value like 0.1) to fine-tune the final accuracy beyond 90%.
Experiment with a slightly higher $\lambda$ (e.g., 0.001) to see if it helps prevent overfitting on the test set.
*/



// Assuming enums are defined like this:
std::string lossTypeToString(Network::LossType loss) {
    switch (loss) {
    case Network::LossType::MSE: return "Mean Squared Error";
    case Network::LossType::CROSS_ENTROPY: return "Cross Entropy";
    default: return "Unknown Loss";
    }
}

std::string neuronTypeToString(Network::NeuronType neuron) {
    switch (neuron) {
    case Network::NeuronType::SIGMOID: return "Sigmoid";
    //case Network::NeuronType::RELU: return "ReLU";
    //case Network::NeuronType::TANH: return "Tanh";
    default: return "Unknown Neuron";
    }
}

void printNetworkParams(const std::string& label, const std::vector<int>& sizes, double l2strength,
    Network::LossType loss, Network::NeuronType neuron) {
    std::cout << "=== " << label << " Network Configuration ===\n";
    std::cout << "Layer sizes: ";
    for (size_t i = 0; i < sizes.size(); ++i) {
        std::cout << sizes[i];
        if (i != sizes.size() - 1) std::cout << " -> ";
    }
    std::cout << "\nL2 Strength: " << l2strength << "\n";
    std::cout << "Loss function: " << lossTypeToString(loss) << "\n";
    std::cout << "Neuron type: " << neuronTypeToString(neuron) << "\n\n";
}

int main() {
    std::vector<int> sizes = { 784, 30, 10 };
    std::string train_images = "data/train-images-idx3-ubyte";
    std::string train_labels = "data/train-labels-idx1-ubyte";
    std::string test_images = "data/t10k-images-idx3-ubyte";
    std::string test_labels = "data/t10k-labels-idx1-ubyte";

    auto training_data = load_mnist_training(train_images, train_labels, 5000);
    auto test_data = load_mnist_test(test_images, test_labels, 3000);

    CPUComputationContext cpuContext;
    GPUComputationContext gpuContext;

    double l2strength = 0.001;
    Network::LossType loss = Network::LossType::CROSS_ENTROPY;
    Network::NeuronType neuron = Network::NeuronType::SIGMOID;

    Network netCPU(sizes, l2strength, loss, neuron, &cpuContext);
    Network netGPU(sizes, l2strength, loss, neuron, &gpuContext);

    int epochs = 5;
    int mini_batch_size = 32;
    double eta = 1.5;

    // Display network parameters
    //printNetworkParams("CPU", sizes, l2strength, loss, neuron);
    printNetworkParams("GPU", sizes, l2strength, loss, neuron);

    /*std::cout << "Training with Cpu context...\n";
    auto cpu_start = std::chrono::high_resolution_clock::now();
    netCPU.SGD(training_data, epochs, mini_batch_size, eta, &test_data, true);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_duration = cpu_end - cpu_start;
    std::cout << "CPU training completed in " << cpu_duration.count() << " seconds.\n";*/

    std::cout << "Training with Gpu context...\n";
    auto gpu_start = std::chrono::high_resolution_clock::now();
    netGPU.SGD(training_data, epochs, mini_batch_size, eta, &test_data, true);
    auto gpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_duration = gpu_end - gpu_start;
    std::cout << "GPU training completed in " << gpu_duration.count() << " seconds.\n";

    return 0;
}



int main_555() {

    // Default parameters
    NeuralNetworkTest tester;
    bool all_passed = tester.runAllTests();

    return 0;
}
