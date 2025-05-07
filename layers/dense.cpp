
#include <xtensor/xrandom.hpp>
#include <xtensor-blas/xlinalg.hpp>

#include "dense.h"
#include "../utilities.h"


Dense::Dense(const std::vector<std::size_t>& p_input_size, size_t p_num_neurons, ActivationID p_activation_id) {
    inp_neurons = p_input_size.back();
    out_neurons = p_num_neurons;

    input_size = p_input_size;
    output_size = p_input_size;
    output_size[output_size.size() - 1] = out_neurons;

    float stddev = std::sqrt(2.0f / static_cast<float>(inp_neurons + out_neurons));
    if (activation_id == ActivationID::RELU) stddev = std::sqrt(2.0f / static_cast<float>(inp_neurons));

    weights = stddev * xt::random::randn<float>({out_neurons, inp_neurons});
    biases = xt::zeros<float>({out_neurons});

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

    num_params = out_neurons * inp_neurons + out_neurons;
}

xt::xarray<float> Dense::feedforward(const xt::xarray<float>& inputs, Mode mode) {
    assert(inputs.shape().back() == inp_neurons);

    input_activations = inputs;

    // actual dense layer feedforward
    outputs = fast_dot(input_activations, xt::transpose(weights)) + biases;
    activations = activation_function(outputs);

    return activations;
}

xt::xarray<float> Dense::backprop(const xt::xarray<float>& p_delta, bool calc_delta_activation) {
    assert(p_delta.shape().back() == out_neurons);

    size_t batch_size = input_activations.shape()[0];
    xt::xtensor<float, 2> delta = p_delta + res_delta;

    if (calc_delta_activation) {
        delta = delta * activation_derivative(outputs);
    }

    grad_weights += fast_dot(xt::transpose(delta), input_activations);
    grad_biases += xt::sum(delta, {0});

    grad_weights /= batch_size;
    grad_biases /= batch_size;

    delta = fast_dot(delta, weights);

    assert(weights.shape() == grad_weights.shape());
    assert(biases.shape() == grad_biases.shape());

    return delta;
}

void Dense::update(float lr) {
    weights -= lr * grad_weights;
    biases -= lr * grad_biases;

    grad_weights.fill(0);
    grad_biases.fill(0);
}

void Dense::update_adam(float lr, float beta1, float beta2) {
    timestep++;

    update_adam_2d(weights, grad_weights, m_weights, v_weights, lr, beta1, beta2, timestep);
    update_adam_1d(biases, grad_biases, m_biases, v_biases, lr, beta1, beta2, timestep);
}

void Dense::update_adamw(float lr, float beta1, float beta2, float weight_decay) {
    timestep++;

    update_adamw_2d(weights, grad_weights, m_weights, v_weights, lr, beta1, beta2, weight_decay, timestep);
    update_adam_1d(biases, grad_biases, m_biases, v_biases, lr, beta1, beta2, timestep);  // no weight decay for biases
}
