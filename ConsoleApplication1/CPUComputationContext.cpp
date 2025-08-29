#include "CPUComputationContext.hpp"
#include<iostream>
#include"utils.h"

void CPUComputationContext::fuckingHell() {
	return;
}

Eigen::VectorXd CPUComputationContext::computeLinearCPU(const Eigen::MatrixXd& weights, const Eigen::VectorXd& input, const Eigen::VectorXd& biases)
{
	// Perform matrix-vector multiplication followed by vector addition
	return weights * input + biases;
}

Eigen::MatrixXd CPUComputationContext::computeWeightGradientCPU(const Eigen::VectorXd& delta, const Eigen::VectorXd& activation)
{
	// Compute outer product: delta (mx1) * activation.transpose() (1xn) = mxn matrix
	return delta * activation.transpose();
}


Eigen::VectorXd CPUComputationContext::applyActivationCPU(const Eigen::VectorXd& z, const Activation* activation)
{
	// Delegate to the activation object's activate method
	return activation->activate(z);
}

Eigen::VectorXd CPUComputationContext::computeActivationDerivativeCPU(const Eigen::VectorXd& activations, const Eigen::VectorXd& pre_activations, const Activation* activation)
{
	// Delegate to the activation object's derivative method
	return activation->derivative(&activations, &pre_activations);
}

//void CPUComputationContext::computeGradientsCPU(const Eigen::VectorXd& deltas, const Eigen::VectorXd& activation_derives, const Eigen::VectorXd& input, Eigen::MatrixXd& weight_grads, Eigen::VectorXd& bias_grads)
//{
//	// Element-wise multiplication of deltas and activation derivatives
//	Eigen::VectorXd adjusted_deltas = deltas.cwiseProduct(activation_derives);
//	// Outer product to compute weight gradients
//	weight_grads = adjusted_deltas * input.transpose();
//	// Bias gradients are the adjusted deltas
//	bias_grads = adjusted_deltas;
//}

void CPUComputationContext::computeGradientsCPU(const Eigen::VectorXd& deltas, const Eigen::VectorXd& activation_derives, const Eigen::VectorXd& input, Eigen::MatrixXd& weight_grads, Eigen::VectorXd& bias_grads, bool apply_derivative)
{
	Eigen::VectorXd adjusted_deltas = deltas;
	if (apply_derivative) {
		// Element-wise multiplication of deltas and activation derivatives
		adjusted_deltas = deltas.cwiseProduct(activation_derives);
	}
	// Outer product to compute weight gradients
	weight_grads = adjusted_deltas * input.transpose();
	// Bias gradients are the adjusted deltas
	bias_grads = adjusted_deltas;
}


void CPUComputationContext::updateParametersCPU(Eigen::MatrixXd& weights, Eigen::VectorXd& biases, const Eigen::MatrixXd& weight_grads, const Eigen::VectorXd& bias_grads, double scale)
{
	// Subtract gradients from weights and biases
	/*weights -= weight_grads;
	biases -= bias_grads;*/

	weights -= weight_grads * scale;
	biases -= bias_grads * scale;
}

void CPUComputationContext::accumulateGradientsCPU(const std::vector<Eigen::MatrixXd>& weight_grads_in, const std::vector<Eigen::VectorXd>& bias_grads_in, std::vector<Eigen::MatrixXd>& weight_grads_out, std::vector<Eigen::VectorXd>& bias_grads_out, double scale)
{
	if (weight_grads_in.size() != weight_grads_out.size() || bias_grads_in.size() != bias_grads_out.size()) {
		throw std::runtime_error("Gradient vector size mismatch in accumulateGradients");
	}
	for (size_t i = 0; i < weight_grads_in.size(); ++i) {
		if (weight_grads_out[i].size() == 0) {
			weight_grads_out[i] = Eigen::MatrixXd::Zero(weight_grads_in[i].rows(), weight_grads_in[i].cols());
		}
		if (bias_grads_out[i].size() == 0) {
			bias_grads_out[i] = Eigen::VectorXd::Zero(bias_grads_in[i].size());
		}
		weight_grads_out[i] += scale * weight_grads_in[i];
		bias_grads_out[i] += scale * bias_grads_in[i];
	}
}

double CPUComputationContext::compute_squared_normCPU(const Eigen::MatrixXd& matrix)
{
	return matrix.squaredNorm();
}

double CPUComputationContext::compute_squared_normCPU(const Eigen::VectorXd& vector)
{
	return vector.squaredNorm();
}


double CPUComputationContext::compute_mse_lossCPU(const Eigen::VectorXd& output, const Eigen::VectorXd& target)
{
	Eigen::VectorXd diff = output - target;
	return diff.squaredNorm();
}

double CPUComputationContext::compute_cross_entropy_lossCPU(const Eigen::VectorXd& output, const Eigen::VectorXd& target)
{
	double loss = 0.0;
	for (int i = 0; i < output.size(); ++i) {
		double a = std::max(1e-15, std::min(1.0 - 1e-15, output(i)));
		loss += -(target(i) * std::log(a) + (1 - target(i)) * std::log(1 - a));
	}
	return loss;
}



