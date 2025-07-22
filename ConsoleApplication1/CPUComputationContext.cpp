#include "CPUComputationContext.hpp"

Eigen::VectorXd CPUComputationContext::computeLinear(const Eigen::MatrixXd& weights, const Eigen::VectorXd& input, const Eigen::VectorXd& biases)
{
	// Perform matrix-vector multiplication followed by vector addition
	return weights * input + biases;
}

Eigen::VectorXd CPUComputationContext::applyActivation(const Eigen::VectorXd& z, const Activation* activation)
{
	// Delegate to the activation object's activate method
	return activation->activate(z);
}

Eigen::VectorXd CPUComputationContext::computeActivationDerivative(const Eigen::VectorXd& activations, const Eigen::VectorXd& pre_activations, const Activation* activation)
{
	// Delegate to the activation object's derivative method
	return activation->derivative(&activations, &pre_activations);
}

void CPUComputationContext::computeGradients(const Eigen::VectorXd& deltas, const Eigen::VectorXd& activation_derives, const Eigen::VectorXd& input, Eigen::MatrixXd& weight_grads, Eigen::VectorXd& bias_grads)
{
	// Element-wise multiplication of deltas and activation derivatives
	Eigen::VectorXd adjusted_deltas = deltas.cwiseProduct(activation_derives);
	// Outer product to compute weight gradients
	weight_grads = adjusted_deltas * input.transpose();
	// Bias gradients are the adjusted deltas
	bias_grads = adjusted_deltas;
}

void CPUComputationContext::updateParameters(Eigen::MatrixXd& weights, Eigen::VectorXd& biases, const Eigen::MatrixXd& weight_grads, const Eigen::VectorXd& bias_grads, double scale)
{
	// Subtract gradients from weights and biases
	/*weights -= weight_grads;
	biases -= bias_grads;*/

	weights -= weight_grads * scale;
	biases -= bias_grads * scale;
}
