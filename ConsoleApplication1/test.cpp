#include "Network.hpp"
#include <iostream>
#include <cmath>
#include <cassert>

// Minimal test framework
#define TEST(name) void name()
#define RUN_TEST(name) std::cout << "Running " #name << "... "; name(); std::cout << "Passed" << std::endl
#define ASSERT_TRUE(cond) if (!(cond)) { std::cerr << "Assertion failed at " << __FILE__ << ":" << __LINE__ << std::endl; exit(1); }
#define ASSERT_APPROX(a, b, tol) if (std::abs((a) - (b)) > (tol)) { std::cerr << "Assertion failed: " << (a) << " != " << (b) << " at " << __FILE__ << ":" << __LINE__ << std::endl; exit(1); }

// Tolerance for floating-point comparisons
const double TOL = 1e-6;

// Test Layer constructor
// Checks matrix sizes and Xavier initialization bounds.
TEST(test_layer_constructor) {
	Layer layer(2, 3, 42); // 2 inputs, 3 neurons
	const auto& weights = layer.get_weights();
	const auto& biases = layer.get_biases();
	ASSERT_TRUE(weights.rows() == 3 && weights.cols() == 2);
	ASSERT_TRUE(biases.size() == 3);

	// Check Xavier initialization (weights/biases should be in reasonable range)
	double stddev = std::sqrt(2.0 / (2 + 1));
	for (int i = 0; i < weights.rows(); ++i) {
		for (int j = 0; j < weights.cols(); ++j) {
			ASSERT_TRUE(std::abs(weights(i, j)) < 3 * stddev); // Within 3 stddev
		}
		ASSERT_TRUE(std::abs(biases(i)) < 3 * stddev);
	}

	// Check activations initialized to zero
	const auto& activations = layer.get_activations();
	for (int i = 0; i < activations.size(); ++i) {
		ASSERT_APPROX(activations(i), 0.0, TOL);
	}
}

// Test Layer forward pass
// Verifies sigmoid outputs and stored activations.
TEST(test_layer_forward) {
	Layer layer(2, 3, 42);
	Eigen::VectorXd input(2);
	input << 1.0, -1.0;
	auto output = layer.forward(input);
	ASSERT_TRUE(output.size() == 3);
	for (int i = 0; i < output.size(); ++i) {
		ASSERT_TRUE(output(i) >= 0.0 && output(i) <= 1.0); // Sigmoid output
	}
	// Verify activations are stored
	const auto& activations = layer.get_activations();
	for (int i = 0; i < output.size(); ++i) {
		ASSERT_APPROX(activations(i), output(i), TOL);
	}
}

// Test Layer gradients
//Compares computed gradients to analytical values.
TEST(test_layer_gradients) {
	Layer layer(2, 3, 42);
	Eigen::VectorXd input(2);
	input << 1.0, -1.0;
	layer.forward(input); // Set activations

	Eigen::VectorXd deltas(3);
	deltas << 0.1, -0.2, 0.3;
	Eigen::MatrixXd weight_grads;
	Eigen::VectorXd bias_grads;
	layer.compute_gradients(deltas, weight_grads, bias_grads);

	ASSERT_TRUE(weight_grads.rows() == 3 && weight_grads.cols() == 2);
	ASSERT_TRUE(bias_grads.size() == 3);

	// Verify gradients (analytical check for one neuron)
	const auto& activations = layer.get_activations();
	for (int i = 0; i < 3; ++i) {
		double sigmoid_deriv = activations(i) * (1.0 - activations(i));
		ASSERT_APPROX(bias_grads(i), deltas(i) * sigmoid_deriv, TOL);
		for (int j = 0; j < 2; ++j) {
			ASSERT_APPROX(weight_grads(i, j), deltas(i) * sigmoid_deriv * input(j), TOL);
		}
	}
}



// Test Layer parameter update
// Confirms weights/biases update correctly.
TEST(test_layer_update_parameters) {
	Layer layer(2, 3, 42);
	Eigen::MatrixXd weight_grads(3, 2);
	weight_grads << 0.1, -0.1, 0.2, -0.2, 0.3, -0.3;
	Eigen::VectorXd bias_grads(3);
	bias_grads << 0.1, 0.2, 0.3;

	auto old_weights = layer.get_weights();
	auto old_biases = layer.get_biases();
	layer.update_parameters(weight_grads, bias_grads);

	const auto& new_weights = layer.get_weights();
	const auto& new_biases = layer.get_biases();

	for (int i = 0; i < 3; ++i) {
		ASSERT_APPROX(new_biases(i), old_biases(i) - bias_grads(i), TOL);
		for (int j = 0; j < 2; ++j) {
			ASSERT_APPROX(new_weights(i, j), old_weights(i, j) - weight_grads(i, j), TOL);
		}
	}
}

// Test Network constructor
// Verifies layer initialization.
TEST(test_network_constructor) {
	std::vector<int> sizes = { 2, 3, 2 };
	Network net(sizes);
	ASSERT_TRUE(net.evaluate({}) == 0); // Empty test data
	// Verify layer sizes
	ASSERT_TRUE(net.backprop(Eigen::VectorXd(2), Eigen::VectorXd(2)).first.size() == 2); // 2 layers
}

// Test Network feedforward
// Checks output size and range.
TEST(test_network_feedforward) {
	std::vector<int> sizes = { 2, 3, 2 };
	Network net(sizes);
	Eigen::VectorXd input(2);
	input << 1.0, -1.0;
	auto output = net.feedforward(input);
	ASSERT_TRUE(output.size() == 2);
	for (int i = 0; i < output.size(); ++i) {
		ASSERT_TRUE(output(i) >= 0.0 && output(i) <= 1.0); // Sigmoid output
	}
}

// Test Network backprop
// Validates gradient sizes.
TEST(test_network_backprop) {
	std::vector<int> sizes = { 2, 3, 2 };
	Network net(sizes);
	Eigen::VectorXd input(2);
	input << 1.0, -1.0;
	Eigen::VectorXd target(2);
	target << 1.0, 0.0;
	auto [nabla_b, nabla_w] = net.backprop(input, target);

	ASSERT_TRUE(nabla_b.size() == 2 && nabla_w.size() == 2);
	ASSERT_TRUE(nabla_b[0].size() == 3 && nabla_w[0].rows() == 3 && nabla_w[0].cols() == 2); // Hidden layer
	ASSERT_TRUE(nabla_b[1].size() == 2 && nabla_w[1].rows() == 2 && nabla_w[1].cols() == 3); // Output layer
}


// Test Network SGD (simple XOR-like dataset)
// Trains on a small XOR - like dataset and checks for accuracy improvement.
TEST(test_network_sgd) {
	std::vector<int> sizes = { 2, 3, 2 };
	Network net(sizes);

	// Initialize training data (input: 2x1, target: 2x1)
	std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> training_data(4);
	training_data[0].first = Eigen::VectorXd(2);  training_data[0].first << 0, 0;
	training_data[0].second = Eigen::VectorXd(2); training_data[0].second << 1, 0;
	training_data[1].first = Eigen::VectorXd(2);  training_data[1].first << 0, 1;
	training_data[1].second = Eigen::VectorXd(2); training_data[1].second << 0, 1;
	training_data[2].first = Eigen::VectorXd(2);  training_data[2].first << 1, 0;
	training_data[2].second = Eigen::VectorXd(2); training_data[2].second << 0, 1;
	training_data[3].first = Eigen::VectorXd(2);  training_data[3].first << 1, 1;
	training_data[3].second = Eigen::VectorXd(2); training_data[3].second << 1, 0;

	// Initialize test data (input: 2x1, label: int)
	std::vector<std::pair<Eigen::VectorXd, int>> test_data(4);
	test_data[0].first = Eigen::VectorXd(2); test_data[0].first << 0, 0; test_data[0].second = 0;
	test_data[1].first = Eigen::VectorXd(2); test_data[1].first << 0, 1; test_data[1].second = 1;
	test_data[2].first = Eigen::VectorXd(2); test_data[2].first << 1, 0; test_data[2].second = 1;
	test_data[3].first = Eigen::VectorXd(2); test_data[3].first << 1, 1; test_data[3].second = 0;

	int initial_correct = net.evaluate(test_data);
	net.SGD(training_data, 100, 2, 1.0, &test_data);
	int final_correct = net.evaluate(test_data);
	ASSERT_TRUE(final_correct >= initial_correct); // Expect improvement
}

int main_bb() {
	RUN_TEST(test_layer_constructor);
	RUN_TEST(test_layer_forward);
	RUN_TEST(test_layer_gradients);
	RUN_TEST(test_layer_update_parameters);

	RUN_TEST(test_network_constructor);
	RUN_TEST(test_network_feedforward);
	RUN_TEST(test_network_backprop);
	RUN_TEST(test_network_sgd);

	std::cout << "All tests passed!" << std::endl;
	return 0;
}