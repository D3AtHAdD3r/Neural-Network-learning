#include "network.h"
#include <iomanip>
#include"utils.h"

Network::Network(const std::vector<int>& sizes) : sizes(sizes), num_layers(sizes.size()), rng(std::random_device{}()) {
    // Initialize biases and weights with Gaussian distribution (mean 0, std 1)
    std::normal_distribution<double> dist(0.0, 1.0);
    for (size_t i = 1; i < sizes.size(); ++i) {
        biases.emplace_back(sizes[i]);
        weights.emplace_back(sizes[i], sizes[i - 1]);
        for (int j = 0; j < sizes[i]; ++j) {
            biases.back()(j) = dist(rng);
            for (int k = 0; k < sizes[i - 1]; ++k) {
                weights.back()(j, k) = dist(rng);
            }
        }
    }
}

Eigen::VectorXd Network::feedforward(const Eigen::VectorXd& a) {
    Eigen::VectorXd activation = a;
    for (size_t i = 0; i < biases.size(); ++i) {
        activation = sigmoid(weights[i] * activation + biases[i]);
    }
    return activation;
}

void Network::SGD(std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>& training_data,
    int epochs, int mini_batch_size, double eta,
    const std::vector<std::pair<Eigen::VectorXd, int>>* test_data) {
    size_t n = training_data.size();
    size_t n_test = test_data ? test_data->size() : 0;
    for (int j = 0; j < epochs; ++j) {
        std::shuffle(training_data.begin(), training_data.end(), rng);
        for (size_t k = 0; k < n; k += mini_batch_size) {
            std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> mini_batch(
                training_data.begin() + k,
                training_data.begin() + std::min(k + mini_batch_size, n));
            update_mini_batch(mini_batch, eta);
        }
        if (test_data) {
            std::cout << "Epoch " << j << ": (Correct Predictions) " << evaluate(*test_data) << " / " << n_test << std::endl;
        }
        else {
            std::cout << "Epoch " << j << " complete" << std::endl;
        }
    }
}

// Updates the network's weights and biases for a single mini-batch using gradient descent.
// mini_batch: A vector of (input, label) pairs, e.g., 10 MNIST images (784x1) and their one-hot labels (10x1).
// eta: Learning rate, controls the step size of updates (e.g., 3.0).
void Network::update_mini_batch(const std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>& mini_batch, double eta) {
    // Initialize gradient accumulators for biases and weights, starting at zero.
    // nabla_b matches biases (e.g., 30x1 for hidden, 10x1 for output).
    // nabla_w matches weights (e.g., 30x784 for input to hidden, 10x30 for hidden to output).
    std::vector<Eigen::VectorXd> nabla_b(biases.size());
    std::vector<Eigen::MatrixXd> nabla_w(weights.size());
    for (size_t i = 0; i < biases.size(); ++i) {
        // Set each bias gradient to a zero vector of the same size as the corresponding bias.
        nabla_b[i] = Eigen::VectorXd::Zero(biases[i].size());
        // Set each weight gradient to a zero matrix of the same size as the corresponding weight matrix.
        nabla_w[i] = Eigen::MatrixXd::Zero(weights[i].rows(), weights[i].cols());
    }

    // Process each training example in the mini-batch.
    // For MNIST, each pair (x, y) is an image (784x1) and its one-hot label (10x1, e.g., [0,0,0,0,0,1,0,0,0,0] for digit 5).
    int idx = 0;
    for (const auto& [x, y] : mini_batch) {
        // Compute gradients for this example using backpropagation.
        // backprop returns (delta_nabla_b, delta_nabla_w), the gradients of the cost function
        // with respect to biases and weights for this single example.
        auto [delta_nabla_b, delta_nabla_w] = backprop(x, y);
       /* std::cout << "Example " << idx << " input norm: " << x.norm() << std::endl;
        idx++;
        ShowGrads(delta_nabla_b, delta_nabla_w, 10, 30, 40);*/
       

        // Accumulate gradients by adding this example's gradients to the running sum.
        // This sums the contribution of each example to the total gradient for the mini-batch.
        for (size_t i = 0; i < biases.size(); ++i) {
            nabla_b[i] += delta_nabla_b[i]; // Add bias gradients.
            nabla_w[i] += delta_nabla_w[i]; // Add weight gradients.
        }

        //ShowGrads(nabla_b, nabla_w);
    }

   
    // Update weights and biases using the average gradient over the mini-batch.
    // Scale by eta / mini_batch.size() to compute the step size for gradient descent.
    // For a mini-batch of 10, this averages the gradients and applies the learning rate.
    double scale = eta / mini_batch.size();
    for (size_t i = 0; i < biases.size(); ++i) {
        // Update weights: w = w - (eta/m) * gradient.
        // This moves weights in the direction that reduces the cost function.
        weights[i] -= scale * nabla_w[i];
        // Update biases: b = b - (eta/m) * gradient.
        // This adjusts biases to improve predictions for the mini-batch's digits.
        biases[i] -= scale * nabla_b[i];
    }
}

std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::MatrixXd>> Network::backprop(
    const Eigen::VectorXd& x, const Eigen::VectorXd& y) const{

    //Gradient Initialization
    std::vector<Eigen::VectorXd> nabla_b(biases.size());
    std::vector<Eigen::MatrixXd> nabla_w(weights.size());
    //All gradients are initialized to zero.
    for (size_t i = 0; i < biases.size(); ++i) {
        nabla_b[i] = Eigen::VectorXd::Zero(biases[i].size());
        nabla_w[i] = Eigen::MatrixXd::Zero(weights[i].rows(), weights[i].cols());
    }

    // Feedforward pass: Computes activations and pre-activations for all layers
    // Input: x (input vector)
    // Stores: activations (output of each layer, including input) and zs (pre-activation values)

    // Start with the input vector as the first activation
    Eigen::VectorXd activation = x;
    std::vector<Eigen::VectorXd> activations = { x }; // Store the input as the first activation(first layer is input layer, so input = activatin)
    std::vector<Eigen::VectorXd> zs;    // Store pre-activation values (z = W*a + b) for each layer

    // Iterate through each layer (excluding input layer)
    for (size_t i = 0; i < biases.size(); ++i) {
        // Compute pre-activation: z = W * a + b
        // weights[i]: Weight matrix connecting layer i to layer i+1
        // activation: Output (activations) of the previous layer
        // biases[i]: Bias vector for layer i+1
        Eigen::VectorXd z = weights[i] * activation + biases[i];
        zs.push_back(z); // Store z for use in backpropagation
        activation = sigmoid(z); // Apply sigmoid activation function to get new activations
        activations.push_back(activation); // Store the new activations for the current layer
    }

    //debug
    /*std::cout << "----->Activations[0]:  \n";
    displayVectorXd(activations[0], 500);
    std::cout << "\n";
    std::cout << "----->Activations[1]:  \n";
    displayVectorXd(activations[1], 500);
    std::cout << "\n";*/

    // Backward pass: Compute gradients for biases and weights

    // Step 1:Compute error (delta) for the output layer.
    // ***In Notes(BackPropogation simplified): delta = ((a-y).σ'(z)). 
    // cost_derivative(activations.back(), y) = (a-y) = derivative of cost function, with respect to networks's output.
    // activations.back(): The output layer's activations, computed in the feedforward pass. y: The target output.
    // sigmoid_prime(zs.back()) = σ'(z) = the derivative of the sigmoid function at z.
    // zs.back() = The pre-activation values for the output layer.
    // cwiseProduct: Element-wise multiplication of the cost derivative and sigmoid derivative.
    Eigen::VectorXd delta = cost_derivative(activations.back(), y).cwiseProduct(sigmoid_prime(zs.back()));

    //debug
    /*std::cout << "----->delta for output layer:  \n";
    displayVectorXd(delta);
    std::cout << "\n";*/
    
    //Store the gradients relative to biases of last or output layer.
    //or The gradient of the cost with respect to the output layer biases.
    // ***In Notes (BackPropogation simplified): dc/db = ((a-y).σ'(z)) . 1 = delta
    nabla_b.back() = delta;

    //Computes and Stores the gradient of the cost with respect to the output layer weights
    // ***In Notes (BackPropogation simplified): dc/dw = ((a-y).σ'(z)) . x, where x=inputs(output from hidden layer in this case)
    //activations[activations.size() - 2]: The activations of the second-to-last layer.
    nabla_w.back() = delta * activations[activations.size() - 2].transpose();

    //debug
    /*std::cout << "----->Weigth Gradients for output layer:  \n";
    displayMatrixXd(nabla_w.back());
    std::cout << "\n";*/
    
    //Hidden Layer Error Propagation
    //Propagates the error backward through the hidden layers, computing errors and gradients for each layer from the second-to-last layer to the first hidden layer.
    for (int l = 2; l < num_layers; ++l) {

        //Retrieve Pre-Activation of current layer(l)
        const Eigen::VectorXd& z = zs[zs.size() - l];

        //debug
        /*std::cout << "----->Pre-Activations for hidden layer:  \n";
        displayVectorXd(z);
        std::cout << "\n";*/

        //Compute Sigmoid Derivative
        Eigen::VectorXd sp = sigmoid_prime(z); //σ'(z^l)

        //debug
        /*std::cout << "----->Sigmoid Derivatives of Pre-Activations :  \n";
        displayVectorXd(sp);
        std::cout << "\n";*/

        //Compute error (delta) for the layer(l)
        delta = (weights[weights.size() - l + 1].transpose() * delta).cwiseProduct(sp); // delta^l = (W^(l+1))^T * delta^(l+1) ⊙ σ'(z^l)

        //debug
       /* std::cout << "----->delta for hidden layer:  \n";
        displayVectorXd(delta);
        std::cout << "\n";*/

        nabla_b[nabla_b.size() - l] = delta; // Gradient for biases: ∂C/∂b^l = delta^l
        nabla_w[nabla_w.size() - l] = delta * activations[activations.size() - l - 1].transpose(); // Gradient for weights: ∂C/∂W^l = delta^l * (a^(l-1))^T

        //debug
       /* std::cout << "----->Weigth Gradients for hidden layer:  \n";
        displayMatrixXd(nabla_w[nabla_w.size() - l], 1500);
        std::cout << "\n";*/
    }
    return { nabla_b, nabla_w };
}

// Perform the feedforward pass for a neural network
// Input: x (input vector)
// Output: None (stores activations and zs for later use)
void Network::feedforward_standalone(const Eigen::VectorXd& x)
{
    Eigen::VectorXd activation = x;  // Initialize with input as the first activation
    std::vector<Eigen::VectorXd> activations = { x };  // Store input activation
    std::vector<Eigen::VectorXd> zs;  // Store pre-activation values

    for (size_t i = 0; i < biases.size(); ++i) {  // Loop over layers (hidden + output)
        // Compute pre-activation: z = W * a + b
        Eigen::VectorXd z = weights[i] * activation + biases[i];
        zs.push_back(z);  // Save z for backpropagation
        activation = sigmoid(z);  // Apply sigmoid to get new activations
        activations.push_back(activation);  // Save activations for next layer or backprop
    }
    // Note: activations.back() is the network's output
}

int Network::evaluate(const std::vector<std::pair<Eigen::VectorXd, int>>& test_data) {
    int correct = 0;
    for (const auto& [x, y] : test_data) {
        Eigen::VectorXd output = feedforward(x);
        int predicted = std::distance(output.data(), std::max_element(output.data(), output.data() + output.size()));
        if (predicted == y) ++correct;
    }
    return correct;
}

Eigen::VectorXd Network::cost_derivative(const Eigen::VectorXd& output_activations, const Eigen::VectorXd& y) const{
    return output_activations - y;
}

void Network::display_biases() const {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== Biases ===" << std::endl;
    for (size_t i = 0; i < biases.size(); ++i) {
        std::string layer_name = (i == biases.size() - 1) ? "Output Layer" : "Hidden Layer " + std::to_string(i + 1);
        std::cout << layer_name << " (" << biases[i].size() << " biases):" << std::endl;
        for (int j = 0; j < biases[i].size(); ++j) {
            std::cout << "  b[" << j << "] = " << biases[i](j);
            if (j < biases[i].size() - 1) std::cout << ",";
            if (j == 9 && biases[i].size() > 10) {
                std::cout << " ... (truncated, total " << biases[i].size() << " biases)";
                break;
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

void Network::display_weights() const {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== Weights ===" << std::endl;
    for (size_t i = 0; i < weights.size(); ++i) {
        std::string from_layer = (i == 0) ? "Input Layer" : "Hidden Layer " + std::to_string(i);
        std::string to_layer = (i == weights.size() - 1) ? "Output Layer" : "Hidden Layer " + std::to_string(i + 1);
        std::cout << "From " << from_layer << " to " << to_layer
            << " (" << weights[i].rows() << "x" << weights[i].cols() << " matrix):" << std::endl;
        int max_rows = std::min(static_cast<int>(weights[i].rows()), 5);
        int max_cols = std::min(static_cast<int>(weights[i].cols()), 5);
        for (int r = 0; r < max_rows; ++r) {
            std::cout << "  [";
            for (int c = 0; c < max_cols; ++c) {
                std::cout << weights[i](r, c);
                if (c < max_cols - 1) std::cout << ", ";
            }
            if (max_cols < weights[i].cols()) std::cout << ", ...";
            std::cout << "]" << std::endl;
        }
        if (max_rows < weights[i].rows() || max_cols < weights[i].cols()) {
            std::cout << "  (Truncated, full size: " << weights[i].rows() << "x" << weights[i].cols() << ")" << std::endl;
        }
        std::cout << std::endl;
    }
}

// Displays the gradients computed by backprop for a single training example.
// x: Input vector (e.g., 784x1 MNIST image).
// y: Desired output (e.g., 10x1 one-hot label for a digit).
// Prints bias gradients (e.g., 30x1 for hidden, 10x1 for output) and weight gradients
// (e.g., 30x784 for input to hidden, 10x30 for hidden to output), with truncation for readability.
void Network::display_backprop_gradients(const Eigen::VectorXd& x, const Eigen::VectorXd& y) const {
    // Compute gradients using backprop for the given example.
    auto [nabla_b, nabla_w] = backprop(x, y);

    // Set output format: fixed-point, 4 decimal places for readability.
    std::cout << std::fixed << std::setprecision(4);

    // Display bias gradients.
    std::cout << "=== Bias Gradients (from backprop) ===" << std::endl;
    for (size_t i = 0; i < nabla_b.size(); ++i) {
        // Label the layer (e.g., Hidden Layer 1 or Output Layer).
        std::string layer_name = (i == nabla_b.size() - 1) ? "Output Layer" : "Hidden Layer " + std::to_string(i + 1);
        // Print the number of bias gradients (e.g., 30 for hidden, 10 for output).
        std::cout << layer_name << " (" << nabla_b[i].size() << " bias gradients):" << std::endl;
        // Print each gradient value with its index.
        for (int j = 0; j < nabla_b[i].size(); ++j) {
            std::cout << "  db[" << j << "] = " << nabla_b[i](j);
            if (j < nabla_b[i].size() - 1) std::cout << ",";
            // Truncate after 10 elements to avoid overwhelming output for large vectors (e.g., 30).
            if (j == 9 && nabla_b[i].size() > 10) {
                std::cout << " ... (truncated, total " << nabla_b[i].size() << " bias gradients)";
                break;
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    // Display weight gradients.
    std::cout << "=== Weight Gradients (from backprop) ===" << std::endl;
    for (size_t i = 0; i < nabla_w.size(); ++i) {
        // Label the source and destination layers (e.g., Input to Hidden Layer 1).
        std::string from_layer = (i == 0) ? "Input Layer" : "Hidden Layer " + std::to_string(i);
        std::string to_layer = (i == nabla_w.size() - 1) ? "Output Layer" : "Hidden Layer " + std::to_string(i + 1);
        // Print the matrix dimensions (e.g., 30x784).
        std::cout << "From " << from_layer << " to " << to_layer
            << " (" << nabla_w[i].rows() << "x" << nabla_w[i].cols() << " gradient matrix):" << std::endl;
        // Limit to 5x5 subset for large matrices (e.g., 30x784).
        int max_rows = std::min(static_cast<int>(nabla_w[i].rows()), 5);
        int max_cols = std::min(static_cast<int>(nabla_w[i].cols()), 5);
        // Print each row of the subset.
        for (int r = 0; r < max_rows; ++r) {
            std::cout << "  [";
            // Print each column value.
            for (int c = 0; c < max_cols; ++c) {
                std::cout << nabla_w[i](r, c);
                if (c < max_cols - 1) std::cout << ", ";
            }
            if (max_cols < nabla_w[i].cols()) std::cout << ", ...";
            std::cout << "]" << std::endl;
        }
        // Indicate truncation if the full matrix is larger.
        if (max_rows < nabla_w[i].rows() || max_cols < nabla_w[i].cols()) {
            std::cout << "  (Truncated, full size: " << nabla_w[i].rows() << "x" << nabla_w[i].cols() << ")" << std::endl;
        }
        std::cout << std::endl;
    }
}

Eigen::VectorXd sigmoid(const Eigen::VectorXd& z) {
    return z.unaryExpr([](double x) { return 1.0 / (1.0 + std::exp(-x)); });
}

Eigen::VectorXd sigmoid_prime(const Eigen::VectorXd& z) {
    Eigen::VectorXd sz = sigmoid(z);
    return sz.cwiseProduct(Eigen::VectorXd::Ones(sz.size()) - sz);
}