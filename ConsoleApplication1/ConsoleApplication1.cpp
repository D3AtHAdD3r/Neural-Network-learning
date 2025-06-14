#include "NeuralNetworkTest.hpp"
#include "Network.hpp"
#include "mnistLoader.h"
#include"utils.h"
#include <iostream>

/*Changes made :
1. Added detailed metrics.
2. Implemented L2 Regularization and L2 Scaling.
*/

/*
Notes- 
Lambda Value:
lambda=0.01 may be too high for a small training set (3,000 MNIST samples). Common values for MNIST are 0.0001 to 0.001, scaled by the dataset size.
*/


int main() {
    try {
        // Example: [784, 30, 10] network
        std::vector<int> sizes = { 784, 30, 10 };
        std::string train_images = "data/train-images-idx3-ubyte";
        std::string train_labels = "data/train-labels-idx1-ubyte";
        std::string test_images = "data/t10k-images-idx3-ubyte";
        std::string test_labels = "data/t10k-labels-idx1-ubyte";

        // Load smaller dataset for testing
        auto training_data = load_mnist_training(train_images, train_labels, 10000);
        auto test_data = load_mnist_test(test_images, test_labels, 1000);

        // Train without regularization
        std::cout << "Training without L2 regularization...\n";
        Network net_no_reg(sizes, 0.0);
        net_no_reg.SGD(training_data, 10, 32, 3.0, &test_data, true);

        // Train with L2 regularization
        std::cout << "\nTraining with L2 regularization (lambda = 0.01)...\n";
        Network net_with_reg(sizes, 0.0001);
        net_with_reg.SGD(training_data, 10, 32, 3.0, &test_data, true);

        return 0;
    }
    catch (const std::bad_alloc& e) {
        std::cerr << "Memory allocation error: " << e.what() << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}



int main_2() {

    // Default parameters
    //NeuralNetworkTest tester;
    //bool all_passed = tester.runAllTests();

    // Example with custom parameters
    std::vector<int> custom_sizes = { 2, 3, 2 };
    NeuralNetworkTest custom_tester(3, 4, 123, custom_sizes);

    bool all_passed;
    all_passed = custom_tester.runAllTests();

    if (all_passed) {
        std::cout << "All test suites passed!" << std::endl;
    }
    else {
        std::cerr << "Some tests failed." << std::endl;
        return 1;
    }

    return 0;
}
