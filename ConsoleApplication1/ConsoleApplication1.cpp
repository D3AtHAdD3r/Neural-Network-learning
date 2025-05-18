#include "Layer.hpp"
#include <iostream>



int main() {

	Layer layer(3, 2, 42); // 3 inputs, 2 neurons
	std::cout << layer.print_parameters(false) << std::endl;

	Eigen::VectorXd input(3);
	input << 1.0, 0.5, -1.0;

	Eigen::VectorXd output = layer.forward(input);
	std::cout << "Output: " << output.transpose() << std::endl;
	std::cout << layer.print_parameters(false) << std::endl; // JSON format
	return 0;
}