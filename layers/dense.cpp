
#include <xtensor/xrandom.hpp>
#include <xtensor-blas/xlinalg.hpp>

#include "dense.h"
#include "../utilities.h"


Dense::Dense(const std::vector<std::size_t>& p_input_size, size_t p_num_neurons, ActivationID p_activation_id) {
    inp_neurons = p_input_size.back();
    out_neurons = p_num_neurons;

    input_size = {inp_neurons};
    output_size = {out_neurons};

    weights = xt::random::rand<float>({out_neurons, inp_neurons}, -1.0, 1.0);
    biases = xt::random::rand<float>({out_neurons}, -1.0, 1.0);

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

    extra_dim_prod = prod(p_input_size, 0, p_input_size.size() - 2);
    input_shape = p_input_size;
    input_shape.insert(input_shape.begin(), 1);
    output_shape = input_shape;
    output_shape[output_shape.size() - 1] = out_neurons;
}

xt::xarray<float> Dense::feedforward(const xt::xarray<float>& inputs, bool evaluation_mode) {
    assert(inputs.shape().back() == inp_neurons);

    size_t real_batch_size = inputs.shape()[0];
    batch_size = real_batch_size * extra_dim_prod;
    input_shape[0] = real_batch_size;
    output_shape[0] = real_batch_size;

    // ensure inputs are in shape {batch_size, inp_neurons}
    input_activations = xt::reshape_view(inputs, {batch_size, inp_neurons});

    // actual dense layer feedforward
    outputs = xt::linalg::dot(input_activations, xt::transpose(weights)) + biases;
    activations = activation_function(outputs);

    return xt::reshape_view(activations, output_shape);
}

xt::xarray<float> Dense::backprop(const xt::xarray<float>& p_delta, bool calc_delta_activation) {
    assert(p_delta.shape().back() == out_neurons);

    // ensure delta is in shape {batch_size, out_neurons}
    xt::xtensor<float, 2> delta = xt::reshape_view(xt::eval(p_delta + res_delta), {batch_size, out_neurons});

    if (calc_delta_activation) {
        delta = delta * activation_derivative(outputs);
    }

    grad_weights += xt::linalg::dot(xt::transpose(delta), input_activations);
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
