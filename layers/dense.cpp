
#include <xtensor/xrandom.hpp>
#include <xtensor-blas/xlinalg.hpp>

#include "dense.h"
#include "../utilities.h"


Dense::Dense(const std::vector<std::size_t>& p_input_size, size_t p_num_neurons, ActivationID p_activation_id) {
    input_size = {p_input_size.back()};
    output_size = {p_num_neurons};

    weights = xt::random::rand<float>({p_num_neurons, input_size[0]}, -1.0, 1.0);
    biases = xt::random::rand<float>({p_num_neurons}, -1.0, 1.0);

    grad_weights = xt::zeros_like(weights);
    grad_biases = xt::zeros_like(biases);

    m_weights = xt::zeros_like(weights);
    m_biases = xt::zeros_like(biases);

    v_weights = xt::zeros_like(weights);
    v_biases = xt::zeros_like(biases);

    timestep = 0;

    activation_id = p_activation_id;

    activation_function = get_activation_function(activation_id);
    activation_derivative = get_activation_derivative(activation_id);
}

xt::xarray<float> Dense::feedforward(const xt::xarray<float>& inputs, bool evaluation_mode) {
    assert(inputs.shape().back() == input_size[0]);
    input_activations = inputs;

    // ensure inputs are in shape {batch_size, x}
    auto input_shape = input_activations.shape();
    size_t batch_size = 1;
    for (size_t i = 0; i < input_shape.size() - 1; i++) batch_size *= input_shape[i];
    xt::xtensor<float, 2> inputs_reshaped = xt::reshape_view(inputs, {batch_size, input_shape.back()});

    // actual dense layer feedforward
    outputs = xt::linalg::dot(inputs_reshaped, xt::transpose(weights)) + biases;
    activations = activation_function(outputs);

    // transform input back
    input_shape[input_shape.size() - 1] = output_size[0];
    return xt::reshape_view(activations, input_shape);
}

xt::xarray<float> Dense::backprop(const xt::xarray<float>& p_delta, bool calc_delta_activation) {
    // ensure inputs and delta are in shape {batch_size, x}
    auto input_shape = input_activations.shape();
    size_t batch_size = 1;
    for (size_t i = 0; i < input_shape.size() - 1; i++) batch_size *= input_shape[i];
    xt::xtensor<float, 2> inputs_reshaped = xt::reshape_view(input_activations, {batch_size, input_shape.back()});
    xt::xtensor<float, 2> delta = xt::reshape_view(xt::eval(p_delta + res_delta), {batch_size, output_size[0]});

    if (calc_delta_activation) {
        delta = delta * activation_derivative(outputs);
    }

    grad_weights += xt::linalg::dot(xt::transpose(delta), inputs_reshaped);
    grad_biases += xt::sum(delta, {0});

    grad_weights /= batch_size;
    grad_biases /= batch_size;

    delta = xt::linalg::dot(delta, weights);

    // reshape delta back to original input shape
    xt::xarray<float> next_delta = xt::reshape_view(delta, input_shape);
    return next_delta;
}

void Dense::update(float lr) {
    weights -= lr * grad_weights;
    biases -= lr * grad_biases;

    grad_weights.fill(0);
    grad_biases.fill(0);
}

void Dense::update_adam(float lr, float beta1, float beta2, float epsilon) {
    timestep++;

    update_adam_2d(weights, grad_weights, m_weights, v_weights, lr, beta1, beta2, epsilon, timestep);
    update_adam_1d(biases, grad_biases, m_biases, v_biases, lr, beta1, beta2, epsilon, timestep);
}
