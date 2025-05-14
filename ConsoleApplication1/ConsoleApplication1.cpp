
#include <iostream>
#include "mnistLoader.h"
#include"Network.h"
#include"utils.h"


int main() {
    try {

        // Example: [784, 30, 10] network
        std::vector<int> sizes = { 784, 30, 10 };
        Network net(sizes);

        // Load MNIST (pseudo-code)
        std::string train_images = "data/train-images-idx3-ubyte";
        std::string train_labels = "data/train-labels-idx1-ubyte";
        std::string test_images = "data/t10k-images-idx3-ubyte";
        std::string test_labels = "data/t10k-labels-idx1-ubyte";

        // Load smaller dataset for testing
        auto training_data = load_mnist_training(train_images, train_labels, 10000);
        auto test_data = load_mnist_test(test_images, test_labels, 1000);

        // Train
        net.SGD(training_data,30, 32, 3.0, &test_data);
        return 0;
    }
    catch (const std::bad_alloc& e) {
        std::cerr << "Memory allocation error: " << e.what() << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return 0;
}


int main3() {
    std::vector<int> sizes = { 784, 30, 10 };
    Network net(sizes);

    // Load MNIST (pseudo-code)
    std::string train_images = "data/train-images-idx3-ubyte";
    std::string train_labels = "data/train-labels-idx1-ubyte";
    std::string test_images = "data/t10k-images-idx3-ubyte";
    std::string test_labels = "data/t10k-labels-idx1-ubyte";

    // Load smaller dataset for testing
    auto mini_batch = load_mnist_training(train_images, train_labels, 10);
   

    // Check weights and biases
        net.display_weights();
        net.display_biases();


     //displayMiniBatch(mini_batch);

    // Process mini_batch
        for (size_t i = 0; i < mini_batch.size(); ++i) 
        {
        const auto& [x, y] = mini_batch[i];
        std::cout << "Example " << i << " input norm: " << x.norm() << std::endl;
        auto [delta_nabla_b, delta_nabla_w] = net.backprop(x, y);
        std::cout << "Gradients for example " << i << ":" << std::endl;
        ShowGrads(delta_nabla_b, delta_nabla_w, 10, 20, 200);
        }
    
        return 0;
}

int main2() {
    // Create a network with architecture [784, 30, 10]
    std::vector<int> sizes = { 784, 30, 10 };
    Network net(sizes);

    // Create a mini-batch of 5 dummy examples
    std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> mini_batch;
    for (int i = 0; i < 5; ++i) {
        Eigen::VectorXd x = Eigen::VectorXd::Random(784); // Random input
        Eigen::VectorXd y = Eigen::VectorXd::Zero(10); y(i % 10) = 1.0; // One-hot label
        mini_batch.emplace_back(x, y);
    }

    // Check weights and biases
    net.display_weights();
    net.display_biases();

    displayMiniBatch(mini_batch);

    // Process mini-batch
    for (size_t i = 0; i < mini_batch.size(); ++i) {
        const auto& [x, y] = mini_batch[i];
        std::cout << "Example " << i << " input norm: " << x.norm() << std::endl;
        auto [delta_nabla_b, delta_nabla_w] = net.backprop(x, y);
        std::cout << "Gradients for example " << i << ":" << std::endl;
        ShowGrads(delta_nabla_b, delta_nabla_w, 10, 30, 40);
    }


    return 0;
}