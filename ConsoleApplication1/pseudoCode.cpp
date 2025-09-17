#include "Network.hpp"
#include"utils.h"
#include <iomanip>
#include <numeric> 

void Network::backprop_gpu_batch(const std::vector<Eigen::VectorXd>& batch_targets, double& batch_loss) {

    int batch_size = static_cast<int>(batch_targets.size());
    if (batch_size > allocated_batch_size_) {
        throw std::runtime_error("Batch size exceeds allocated buffers");
    }

    // Copy targets to device (column-major: output_size x batch_size)
    contextGPU_->copy_batch_to_device(d_batch_targets, batch_targets, false);

    int output_size = sizes.back();
    double* d_output_batch = d_batch_activations.back();  // From forward pass

    // Compute output delta : delta_L = output - target(for MSE or sigmoid + CE)
    contextGPU_->launch_elementwise_subtract_batch(d_output_batch, d_batch_targets, d_batch_deltas.back(), output_size, batch_size);

    // Apply activation derivative to output delta
    layers.back()->apply_derivative_gpu_batch(d_batch_deltas.back(), batch_size);

    // Compute batch loss (MSE for now; extend for CE)
    if (loss_type_ == LossType::MSE) {
        batch_loss = contextGPU_->compute_mse_loss_batch_gpu(d_output_batch, d_batch_targets, output_size, batch_size);
    }
    else {
        // Stub for CE: Implement batched CE loss reduction
        batch_loss = 0.0;  // Placeholder
    }


    // Propagate deltas backward (from second-last layer to first)
    for (int l = static_cast<int>(layers.size()) - 2; l >= 0; --l) {
        Layer* next_layer = layers[l + 1].get();

        // delta_l = W_{l+1}^T * delta_{l+1} (input_size_l x batch_size)
        contextGPU_->compute_delta_back_batch(next_layer->get_d_weights(), d_batch_deltas[l + 1],
            d_batch_deltas[l], next_layer->get_num_inputs(),
            next_layer->get_num_neurons(), batch_size);
        // Apply derivative for hidden layer
        layers[l]->apply_derivative_gpu_batch(d_batch_deltas[l], batch_size);
    }

    // Compute gradients (forward order: use prev activations as input for each layer)
    const double* d_prev_a = d_batch_main_input;  // First layer input
    for (size_t l = 0; l < layers.size(); ++l) {
        // No deriv apply here (already done in delta prop); pass nullptr for d_derivatives_batch
        contextGPU_->computeGradientsGPU_batch(d_batch_deltas[l], d_prev_a, nullptr,
            accumulate_weight_grads[l], accumulate_bias_grads[l],
            sizes[l + 1], sizes[l], batch_size, false);
        d_prev_a = d_batch_activations[l];  // Next prev_a
    }

}

//Pseudo code
//Keep structure as close to backprop_gpu as possible
void Network::backprop_gpu_batch(const std::vector<Eigen::VectorXd>& batch_targets, double& batch_loss) {

    int batch_size = static_cast<int>(batch_targets.size());
    if (batch_size > allocated_batch_size_) {
        throw std::runtime_error("Batch size exceeds allocated buffers");
    }

    // Copy targets to device (column-major: output_size x batch_size) in batch_targets
    contextGPU_->copy_batch_to_device(d_batch_targets, batch_targets, false);
    // get batched output activations for last layer
    int output_size = sizes.back();
    double* d_output_batch = d_batch_activations.back();  // From forward pass

    bool apply_deriv = false;


    //calculate batched delta for output layer.
    switch (loss_type_) { // Output layer: compute delta = cost_derivative * (sigmoid' for MSE, 1 for CE)
    case LossType::MSE: {
        apply_deriv = true;
        cost_prime_mse_crossent_batched(d_output_batch, d_batch_targets, d_batch_deltas.back(), output_size, batch_size);
    };
    case LossType::CROSS_ENTROPY: {
        apply_deriv = false;
        cost_prime_mse_crossent_batched(d_output_batch, d_batch_targets, d_batch_deltas.back(), output_size, batch_size);
        break;
    };
    default:
        throw std::runtime_error("Unsupported loss type");
        break;
    };

    // Propagate deltas backward (from second-last layer to first)
    // calculate batched delta

    //for output layer
    //unlike backprop_gpu which applies derivative inside compute_gradients_gpu,
    //we apply the derivative(adjust deltas according to activation derivatives), here only for clarity
    //applying derivative is conditional for first layer. check flag apply_deriv.
    // but for hidden layers we always apply. that's up to my knowledge.

    // Apply activation derivative to output delta
    // how you suggested:
    layers.back()->apply_derivative_gpu_batch(d_batch_deltas.back(), batch_size);
    //layers.back()->apply_derivative_gpu_batch consist of two parts
    //1-computeActivationDerivativeGPU_batch, which needs d_activations, d_pre_activations,d_dy, rows and batch size.
    //all of them can be provided here, since batched activations and pre-activations are already stored in network.
    //so GPUComputationContext::computeActivationDerivativeGPU_batch can be called from here. centralizing the shit.
    //also note: since we have implemented forward_pass_batch_processing, activations and pre-activations are getting stored in network(centralized), not in respective layers.
    //2- contextGPU_->launch_elementwise_multiply_batch. 
    //which can  also be called directly from here.

    // calculate batched delta
    for (int l = static_cast<int>(layers.size()) - 2; l >= 0; --l) {
        Layer* next_layer = layers[l + 1].get();

        // delta_l = W_{l+1}^T * delta_{l+1} (input_size_l x batch_size)
        contextGPU_->compute_delta_back_batch(next_layer->get_d_weights(), d_batch_deltas[l + 1],
            d_batch_deltas[l], next_layer->get_num_inputs(),
            next_layer->get_num_neurons(), batch_size);
        
        //apply derivate directly from here, just like output layer
        //.....
    }

    // Compute gradients (forward order: use prev activations as input for each layer)
    const double* d_prev_a = d_batch_main_input;  // First layer input
    for (size_t l = 0; l < layers.size(); ++l) {
        
        //changes to me made to func contextGPU_->computeGradientsGPU_batch. mentioned later.
        contextGPU_->computeGradientsGPU_batch(...);
    }
}

// wrapper to call launch_elementwise_subtract_batch
void cost_prime_mse_crossent_batched(...) {
    //calls launch_elementwise_subtract_batch(const double* a, const double* b, double* c, int rows, int batch_size)
}

//no need for anything related to derivatives in the func, since  they are getting applied elsewhere. 
//update accordingly.
contextGPU_->computeGradientsGPU_batch() {

}

question:
with the above update, udpate_min_batch() doesnt need to call contextGPU_->accumulateGradientsGPU... seperately, its getting done in backprop_batched, as i think.

balls in your court.







