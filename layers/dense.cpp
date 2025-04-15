
#include <xtensor/xrandom.hpp>
#include <xtensor-blas/xlinalg.hpp>

#include "dense.h"
#include "../utilities.h"


Dense::Dense(const std::vector<std::size_t>& p_input_size, size_t p_num_neurons, ActivationID p_activation_id) {
    input_size = p_input_size;
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
    input_activations = inputs;

    // standard y = Wx, s.t. x, biases, inputs are all column vectors:
    // output = xt::linalg::dot(weights, input_activation) + biases;

    // batching implementation, y = xW^T:
    // x: (batch_size, input_size)
    // W: (output_size, input_size)

    outputs = xt::linalg::dot(input_activations, xt::transpose(weights)) + biases;
    activations = activation_function(outputs);

    return activations;
}

xt::xarray<float> Dense::backprop(const xt::xarray<float>& delta, bool calc_delta_activation) {
    size_t batch_size = input_activations.shape()[0];

    xt::xtensor<float, 2> next_delta = delta;

    if (calc_delta_activation) {
        next_delta = delta * activation_derivative(outputs);
    }

    // x: (batch_size, input_size)
    // W: (output_size, input_size)
    // next_delta: (batch_size, output_size)

    grad_weights += xt::linalg::dot(xt::transpose(next_delta), input_activations);
    grad_biases += xt::sum(next_delta, {0});

    grad_weights /= batch_size;
    grad_biases /= batch_size;

    next_delta = xt::linalg::dot(next_delta, weights);
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
