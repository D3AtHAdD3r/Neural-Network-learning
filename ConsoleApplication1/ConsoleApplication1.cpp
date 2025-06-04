#include "Network.hpp"
#include "mnistLoader.h"
#include"utils.h"
#include <iostream>

/*
Changes Made:
1. Neuron centric design, with added layer class.
*/

int main() {
    try {
        // Example: [784, 30, 10] network
        std::vector<int> sizes = { 784, 30, 10 };
        Network net(sizes);

        //net.display_weights();
        //net.display_biases();
        //net.display_layer_weights(10);
        //net.display_layer_biases(100);

        // Load MNIST (pseudo-code)
        std::string train_images = "data/train-images-idx3-ubyte";
        std::string train_labels = "data/train-labels-idx1-ubyte";
        std::string test_images = "data/t10k-images-idx3-ubyte";
        std::string test_labels = "data/t10k-labels-idx1-ubyte";

        // Load smaller dataset for testing
        auto training_data = load_mnist_training(train_images, train_labels, 3000);
        auto test_data = load_mnist_test(test_images, test_labels, 1000);

        // Train
        net.SGD(training_data, 30, 32, 3.0, &test_data);
        return 0;
    }
    catch (const std::bad_alloc& e) {
        std::cerr << "Memory allocation error: " << e.what() << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    
}



