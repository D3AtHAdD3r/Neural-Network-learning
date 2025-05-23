#pragma once

#ifndef UTILS_H
#define UTILS_H

#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <iomanip>
#include <string>

// Displays bias and weight gradients (e.g., from backprop) for a single training example.
// delta_nabla_b: Bias gradients, e.g., [30x1, 10x1] for hidden and output layers in a [784, 30, 10] network.
// delta_nabla_w: Weight gradients, e.g., [30x784, 10x30] for input-to-hidden and hidden-to-output.
// max_biases: Maximum number of bias gradients to show per layer (truncates the rest).
// max_weight_rows: Maximum number of rows to show for weight gradient matrices.
// max_weight_cols: Maximum number of columns to show for weight gradient matrices.
// Prints gradients with truncation for readability, similar to Network::display_backprop_gradients.
void ShowGrads(const std::vector<Eigen::VectorXd>& delta_nabla_b,
    const std::vector<Eigen::MatrixXd>& delta_nabla_w,
    int max_biases,
    int max_weight_rows,
    int max_weight_cols);


void displayMiniBatch(const std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>& mini_batch);

void displayVectorXd(const Eigen::VectorXd& vec, size_t max_elements = 0);

void displayMatrixXd(const Eigen::MatrixXd& mat, size_t max_elements = 0);

#endif




