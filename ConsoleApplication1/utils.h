#pragma once
//#ifndef UTILS_H
//#define UTILS_H
//
//#include <Eigen/Dense>
//#include <vector>
//#include <iostream>
//#include <iomanip>
//#include <string>
//
//// Displays bias and weight gradients (e.g., from backprop) for a single training example.
//// delta_nabla_b: Bias gradients, e.g., [30x1, 10x1] for hidden and output layers in a [784, 30, 10] network.
//// delta_nabla_w: Weight gradients, e.g., [30x784, 10x30] for input-to-hidden and hidden-to-output.
//// Prints gradients with truncation for readability, similar to Network::display_backprop_gradients.
//void ShowGrads(const std::vector<Eigen::VectorXd>& delta_nabla_b,
//    const std::vector<Eigen::MatrixXd>& delta_nabla_w) {
//    // Set output format: fixed-point, 4 decimal places for readability.
//    std::cout << std::fixed << std::setprecision(4);
//
//    // Display bias gradients.
//    std::cout << "=== Bias Gradients ===" << std::endl;
//    for (size_t i = 0; i < delta_nabla_b.size(); ++i) {
//        // Label the layer generically (e.g., Layer 1 for hidden, Layer 2 for output).
//        std::string layer_name = "Layer " + std::to_string(i + 1) + " Biases";
//        // Print the number of bias gradients (e.g., 30 for hidden, 10 for output).
//        std::cout << layer_name << " (" << delta_nabla_b[i].size() << " bias gradients):" << std::endl;
//        // Print each gradient value with its index.
//        for (int j = 0; j < delta_nabla_b[i].size(); ++j) {
//            std::cout << "  db[" << j << "] = " << delta_nabla_b[i](j);
//            if (j < delta_nabla_b[i].size() - 1) std::cout << ",";
//            // Truncate after 10 elements to avoid overwhelming output for large vectors (e.g., 30).
//            if (j == 9 && delta_nabla_b[i].size() > 10) {
//                std::cout << " ... (truncated, total " << delta_nabla_b[i].size() << " bias gradients)";
//                break;
//            }
//            std::cout << std::endl;
//        }
//        std::cout << std::endl;
//    }
//
//    // Display weight gradients.
//    std::cout << "=== Weight Gradients ===" << std::endl;
//    for (size_t i = 0; i < delta_nabla_w.size(); ++i) {
//        // Label the source and destination layers (e.g., From Layer 0 to Layer 1 for input to hidden).
//        std::string from_layer = "Layer " + std::to_string(i);
//        std::string to_layer = "Layer " + std::to_string(i + 1);
//        // Print the matrix dimensions (e.g., 30x784).
//        std::cout << "From " << from_layer << " to " << to_layer
//            << " (" << delta_nabla_w[i].rows() << "x" << delta_nabla_w[i].cols() << " gradient matrix):" << std::endl;
//        // Limit to 5x5 subset for large matrices (e.g., 30x784).
//        int max_rows = std::min(static_cast<int>(delta_nabla_w[i].rows()), 5);
//        int max_cols = std::min(static_cast<int>(delta_nabla_w[i].cols()), 5);
//        // Print each row of the subset.
//        for (int r = 0; r < max_rows; ++r) {
//            std::cout << "  [";
//            // Print each column value.
//            for (int c = 0; c < max_cols; ++c) {
//                std::cout << delta_nabla_w[i](r, c);
//                if (c < max_cols - 1) std::cout << ", ";
//            }
//            if (max_cols < delta_nabla_w[i].cols()) std::cout << ", ...";
//            std::cout << "]" << std::endl;
//        }
//        // Indicate truncation if the full matrix is larger.
//        if (max_rows < delta_nabla_w[i].rows() || max_cols < delta_nabla_w[i].cols()) {
//            std::cout << "  (Truncated, full size: " << delta_nabla_w[i].rows() << "x" << delta_nabla_w[i].cols() << ")" << std::endl;
//        }
//        std::cout << std::endl;
//    }
//}
//
//#endif


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




