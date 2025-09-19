std::pair<int, double> Network::evaluate_batch(const std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>& test_data, size_t n) {
    int correct = 0;
    double total_loss = 0.0;
    double weight_norm = 0.0;

    if (!is_gpu_context_) {
        throw std::runtime_error("Unsupported computation context type");
    }

    int batch_size = static_cast<int>(test_data.size());
    if (batch_size == 0)
        throw std::runtime_error("test data size zero");

    if (!batch_buffers_allocated_) {
        init_batch_buffers(test_data.size());
    }

    for (const auto& layer : layers) {
        weight_norm += contextGPU_->compute_squared_normGPU(layer->get_weights());
    }

    // Prepare inputs and targets
    std::vector<Eigen::VectorXd> batch_inputs(batch_size);
    std::vector<Eigen::VectorXd> batch_targets(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        batch_inputs[i] = test_data[i].first;
        batch_targets[i] = test_data[i].second;
    }

    // Forward pass
    std::vector<Eigen::VectorXd> batch_outputs;
    feedforward_gpu_batch(batch_inputs, batch_outputs);

    for (int i = 0; i < batch_targets.size(); ++i) {
        if (is_correct_prediction(batch_outputs[i], batch_targets[i]))
            ++correct;

        if (neuron_type_ == NeuronType::SIGMOID && loss_type_ == LossType::MSE) {
            total_loss += contextGPU_->compute_mse_lossGPU(batch_outputs[i], batch_targets[i]);
        }
        else if (neuron_type_ == NeuronType::SIGMOID && loss_type_ == LossType::CROSS_ENTROPY) {
            total_loss += contextGPU_->compute_cross_entropy_lossGPU(batch_outputs[i], batch_targets[i]);
        }
        else {
            throw std::runtime_error("Unsupported loss type");
        }
    }

    if (lambda > 0.0 && n > 0) {
        total_loss += 0.5 * lambda * weight_norm / n;
    }

    last_test_loss = total_loss;
    return { correct, total_loss };
}

std::pair<int, double> Network::evaluate_batch(const std::vector<std::pair<Eigen::VectorXd, int>>& test_data, size_t n) {

    int correct = 0;
    double total_loss = 0.0;
    double weight_norm = 0.0;

    if (!is_gpu_context_) {
        throw std::runtime_error("Unsupported computation context type");
    }

    int batch_size = static_cast<int>(test_data.size());
    if (batch_size == 0)
        throw std::runtime_error("test data size zero");

    if (!batch_buffers_allocated_) {
        init_batch_buffers(test_data.size());
    }

    for (const auto& layer : layers) {
        weight_norm += contextGPU_->compute_squared_normGPU(layer->get_weights());
    }

    // Prepare inputs and targets
    std::vector<Eigen::VectorXd> batch_inputs(batch_size);
    std::vector<Eigen::VectorXd> batch_targets(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        batch_inputs[i] = test_data[i].first;
        Eigen::VectorXd target = Eigen::VectorXd::Zero(sizes.back());
        target(test_data[i].second) = 1.0;
        batch_targets[i] = target;
    }

    // Forward pass
    std::vector<Eigen::VectorXd> batch_outputs;
    feedforward_gpu_batch(batch_inputs, batch_outputs);

    for (int i = 0; i < batch_targets.size(); ++i) {
        if (is_correct_prediction(batch_outputs[i], batch_targets[i]))
            ++correct;

        if (neuron_type_ == NeuronType::SIGMOID && loss_type_ == LossType::MSE) {
            total_loss += contextGPU_->compute_mse_lossGPU(batch_outputs[i], batch_targets[i]);
        }
        else if (neuron_type_ == NeuronType::SIGMOID && loss_type_ == LossType::CROSS_ENTROPY) {
            total_loss += contextGPU_->compute_cross_entropy_lossGPU(batch_outputs[i], batch_targets[i]);
        }
        else {
            throw std::runtime_error("Unsupported loss type");
        }
    }

    if (lambda > 0.0 && n > 0) {
        total_loss += 0.5 * lambda * weight_norm / n;
    }

    last_test_loss = total_loss;
    return { correct, total_loss };

}