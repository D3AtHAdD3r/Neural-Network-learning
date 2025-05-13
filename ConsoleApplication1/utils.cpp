#include"utils.h"

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
    int max_weight_cols) {
    // Set output format: fixed-point, 4 decimal places for readability.
    std::cout << std::fixed << std::setprecision(4);

    // Display bias gradients.
    std::cout << "=== Bias Gradients ===" << std::endl;
    for (size_t i = 0; i < delta_nabla_b.size(); ++i) {
        // Label the layer generically (e.g., Layer 1 for hidden, Layer 2 for output).
        std::string layer_name = "Layer " + std::to_string(i + 1) + " Biases";
        // Print the number of bias gradients (e.g., 30 for hidden, 10 for output).
        std::cout << layer_name << " (" << delta_nabla_b[i].size() << " bias gradients):" << std::endl;
        // Limit to max_biases elements.
        int display_biases = std::min(static_cast<int>(delta_nabla_b[i].size()), max_biases);
        // Print each gradient value with its index.
        for (int j = 0; j < display_biases; ++j) {
            std::cout << "  db[" << j << "] = " << delta_nabla_b[i](j);
            if (j < display_biases - 1) std::cout << ",";
            std::cout << std::endl;
        }
        // Indicate truncation if not all biases are shown.
        if (display_biases < delta_nabla_b[i].size()) {
            std::cout << "  ... (truncated, total " << delta_nabla_b[i].size() << " bias gradients)" << std::endl;
        }
        std::cout << std::endl;
    }

    // Display weight gradients.
    std::cout << "=== Weight Gradients ===" << std::endl;
    for (size_t i = 0; i < delta_nabla_w.size(); ++i) {
        // Label the source and destination layers (e.g., From Layer 0 to Layer 1 for input to hidden).
        std::string from_layer = "Layer " + std::to_string(i);
        std::string to_layer = "Layer " + std::to_string(i + 1);
        // Print the matrix dimensions (e.g., 30x784).
        std::cout << "From " << from_layer << " to " << to_layer
            << " (" << delta_nabla_w[i].rows() << "x" << delta_nabla_w[i].cols() << " gradient matrix):" << std::endl;
        // Limit to max_weight_rows and max_weight_cols.
        int display_rows = std::min(static_cast<int>(delta_nabla_w[i].rows()), max_weight_rows);
        int display_cols = std::min(static_cast<int>(delta_nabla_w[i].cols()), max_weight_cols);
        // Print each row of the subset.
        for (int r = 0; r < display_rows; ++r) {
            std::cout << "  [";
            // Print each column value.
            for (int c = 0; c < display_cols; ++c) {
                std::cout << delta_nabla_w[i](r, c);
                if (c < display_cols - 1) std::cout << ", ";
            }
            if (display_cols < delta_nabla_w[i].cols()) std::cout << ", ...";
            std::cout << "]" << std::endl;
        }
        // Indicate truncation if the full matrix is larger.
        if (display_rows < delta_nabla_w[i].rows() || display_cols < delta_nabla_w[i].cols()) {
            std::cout << "  (Truncated, full size: " << delta_nabla_w[i].rows() << "x" << delta_nabla_w[i].cols() << ")" << std::endl;
        }
        std::cout << std::endl;
    }
}


void displayMiniBatch(const std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>& mini_batch) {
    std::cout << "Mini-batch contents:\n";
    for (size_t i = 0; i < mini_batch.size(); ++i) {
        std::cout << "Sample " << i + 1 << ":\n";
        std::cout << "  Input: [ ";
        for (int j = 0; j < mini_batch[i].first.size(); ++j) {
            std::cout << mini_batch[i].first(j);
            if (j < mini_batch[i].first.size() - 1) std::cout << ", ";
        }
        std::cout << " ]\n";

        std::cout << "  Output: [ ";
        for (int j = 0; j < mini_batch[i].second.size(); ++j) {
            std::cout << mini_batch[i].second(j);
            if (j < mini_batch[i].second.size() - 1) std::cout << ", ";
        }
        std::cout << " ]\n\n";
    }
}

void displayVectorXd(const Eigen::VectorXd& vec, size_t max_elements) {
    std::cout << "VectorXd (size: " << vec.size() << "):\n[ ";

    size_t limit = (max_elements == 0 || max_elements > vec.size()) ? vec.size() : max_elements;
    for (int i = 0; i < limit; ++i) {
        std::cout << vec(i);
        if (i < limit - 1) std::cout << ", ";
    }

    if (limit < vec.size()) std::cout << ", ..."; // Indicate truncation
    std::cout << " ]\n";
}

void displayMatrixXd(const Eigen::MatrixXd& mat, size_t max_elements) {
    std::cout << "MatrixXd (rows: " << mat.rows() << ", cols: " << mat.cols() << "):\n";

    size_t limit = (max_elements == 0 || max_elements > mat.size()) ? mat.size() : max_elements;
    size_t count = 0;

    for (int i = 0; i < mat.rows(); ++i) {
        std::cout << "[ ";
        for (int j = 0; j < mat.cols(); ++j) {
            if (count < limit) {
                std::cout << mat(i, j);
                if (j < mat.cols() - 1) std::cout << ", ";
                ++count;
            }
            else {
                if (j == 0) std::cout << "...";
                break;
            }
        }
        std::cout << " ]\n";
        if (count >= limit && i < mat.rows() - 1) {
            std::cout << "[ ... ]\n";
            break;
        }
    }
    if (count < mat.size()) std::cout << "...\n";
}