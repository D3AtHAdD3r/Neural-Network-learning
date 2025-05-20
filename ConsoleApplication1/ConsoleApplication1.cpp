#include "Layer.hpp"
#include <iostream>

int main() {
    Layer layer(3, 2, 42); // 3 inputs, 2 neurons
    std::cout << "Initial Parameters:\n" << layer.print_parameters(false) << "\n\n";

    Eigen::VectorXd input(3);
    input << 1.0, 0.5, -1.0;
    Eigen::VectorXd output = layer.forward(input);
    std::cout << "Output: " << output.transpose() << "\n\n";

    Eigen::VectorXd deltas(2);
    deltas << 0.1, -0.1;
    Eigen::MatrixXd weight_grads;
    Eigen::VectorXd bias_grads;
    layer.compute_gradients(deltas, weight_grads, bias_grads);
    std::cout << "Weight Gradients:\n" << weight_grads << "\nBias Gradients:\n" << bias_grads.transpose() << "\n\n";

    layer.update_parameters(weight_grads, bias_grads, 0.01);
    std::cout << "Updated Parameters:\n" << layer.print_parameters(false) << "\n";

    return 0;
}