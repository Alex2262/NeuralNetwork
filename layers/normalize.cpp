//
// Created by Alexander Tian on 4/16/25.
//

#include <xtensor-blas/xlinalg.hpp>

#include "normalize.h"
#include "../utilities.h"

Normalize::Normalize(const std::vector<size_t>& p_input_size) {
    input_size = p_input_size;
    output_size = input_size;

    feature_size = input_size.back();

    gamma = xt::ones<float>({feature_size});
    beta = xt::zeros<float>({feature_size});

    grad_gamma = xt::zeros_like(gamma);
    grad_beta = xt::zeros_like(beta);

    m_gamma = xt::zeros_like(gamma);
    m_beta = xt::zeros_like(beta);

    v_gamma = xt::zeros_like(gamma);
    v_beta = xt::zeros_like(beta);
    num_params = feature_size + feature_size;
}

xt::xarray<float> Normalize::feedforward(const xt::xarray<float>& inputs, Mode mode) {
    assert(inputs.shape().back() == feature_size);

    input_activations = inputs;
    size_t batch_size = input_activations.shape()[0];

    mean = xt::reshape_view(xt::sum(input_activations, {1}) / static_cast<float>(feature_size), std::vector<size_t>{batch_size, 1});

    xmu = input_activations - mean;
    variance = xt::reshape_view(xt::sum(xmu * xmu, {1}) / static_cast<float>(feature_size), std::vector<size_t>{batch_size, 1});
    std_inv = 1.0f / xt::sqrt(variance + eps);

    normalized = xmu * std_inv;

    outputs = normalized * gamma + beta;

    return outputs;
}

xt::xarray<float> Normalize::backprop(const xt::xarray<float>& p_delta, bool calc_delta_activation) {
    assert(p_delta.shape().back() == feature_size);

    size_t batch_size = input_activations.shape()[0];
    xt::xtensor<float, 2> delta = p_delta + res_delta;

    grad_gamma += xt::sum(delta * normalized, {0});
    grad_beta += xt::sum(delta, {0});

    xt::xtensor<float, 2> dxhat = delta * gamma;

    xt::xtensor<float, 2> sum_dxhat = xt::reshape_view(xt::sum(dxhat, {1}), std::vector<size_t>{batch_size, 1});
    xt::xtensor<float, 2> dot_dxhat = xt::reshape_view(xt::sum(dxhat * normalized, {1}), std::vector<size_t>{batch_size, 1});

    xt::xtensor<float, 2> dx = (dxhat
            - sum_dxhat / static_cast<float>(feature_size)
            - normalized * dot_dxhat / static_cast<float>(feature_size)) * std_inv;

    grad_gamma /= batch_size;
    grad_beta /= batch_size;

    return dx;
}

void Normalize::update(float lr) {
    gamma -= lr * grad_gamma;
    beta -= lr * grad_beta;

    grad_gamma.fill(0);
    grad_beta.fill(0);
}

void Normalize::update_adam(float lr, float beta1, float beta2, size_t timestep) {
    timestep++;

    update_adam_1d(gamma, grad_gamma, m_gamma, v_gamma, lr, beta1, beta2, timestep);
    update_adam_1d(beta, grad_beta, m_beta, v_beta, lr, beta1, beta2, timestep);
}

void Normalize::update_adamw(float lr, float beta1, float beta2, float weight_decay, size_t timestep) {
    update_adam(lr, beta1, beta2, timestep);
}

void Normalize::save_weights(std::vector<float>& all) {
    save_1d(all, gamma);
    save_1d(all, m_gamma);
    save_1d(all, v_gamma);

    save_1d(all, beta);
    save_1d(all, m_beta);
    save_1d(all, v_beta);
}

void Normalize::load_weights(xt::xtensor<float, 1>& all, size_t& index) {
    load_1d(all, gamma, index);
    load_1d(all, m_gamma, index);
    load_1d(all, v_gamma, index);

    load_1d(all, beta, index);
    load_1d(all, m_beta, index);
    load_1d(all, v_beta, index);
}
