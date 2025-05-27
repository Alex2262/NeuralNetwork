
#include <xtensor/generators/xrandom.hpp>
#include <xtensor-blas/xlinalg.hpp>

#include "convolution.h"
#include "../utilities.h"


Convolution::Convolution(std::vector<size_t>& p_input_size, size_t p_num_filters, size_t p_filter_size, size_t p_stride,
                          ActivationID p_activation_id) {

    input_size = p_input_size;
    num_filters = p_num_filters;
    filter_size = p_filter_size;
    stride = p_stride;

    size_t out_h = (input_size[0] - filter_size) / stride + 1;
    size_t out_w = (input_size[1] - filter_size) / stride + 1;

    output_size = {out_h, out_w, num_filters};

    size_t amt = filter_size * filter_size * input_size[2];
    if (activation_id != ActivationID::RELU) amt += filter_size * filter_size * num_filters;

    float stddev = std::sqrt(2.0f / static_cast<float>(amt));
    weights = stddev * xt::random::randn<float>({num_filters, filter_size, filter_size, input_size[2]});
    biases = xt::zeros<float>({num_filters});

    grad_weights = xt::zeros_like(weights);
    grad_biases = xt::zeros_like(biases);

    m_weights = xt::zeros_like(weights);
    m_biases = xt::zeros_like(biases);

    v_weights = xt::zeros_like(weights);
    v_biases = xt::zeros_like(biases);

    activation_id = p_activation_id;

    activation_function = get_activation_function(activation_id);
    activation_derivative = get_activation_derivative(activation_id);

    num_params = num_filters * filter_size * filter_size * input_size[2] + num_filters;
}


xt::xarray<float> Convolution::feedforward(const xt::xarray<float>& inputs, Mode mode) {
    input_activations = inputs;
    size_t batch_size = input_activations.shape()[0];

    outputs = xt::zeros<float>({batch_size, output_size[0], output_size[1], output_size[2]});

    for (size_t b = 0; b < batch_size; b++) {
        for (size_t k = 0; k < num_filters; k++) {
            for (size_t i = 0; i < output_size[0]; i++) {
                for (size_t j = 0; j < output_size[1]; j++) {

                    // Take sum of convolutions
                    float sum = 0;

                    for (size_t fi = 0; fi < filter_size; fi++) {
                        for (size_t fj = 0; fj < filter_size; fj++) {
                            size_t in_i = i * stride + fi;
                            size_t in_j = j * stride + fj;
                            for (size_t c = 0; c < input_size[2]; c++) {
                                sum += weights(k, fi, fj, c) * input_activations(b, in_i, in_j, c);
                            }
                        }
                    }

                    outputs(b, i, j, k) = sum + biases(k);
                }
            }
        }
    }


    activations = activation_function(outputs);
    return activations;
}

xt::xarray<float> Convolution::backprop(const xt::xarray<float>& p_delta, bool calc_delta_activation) {
    size_t batch_size = input_activations.shape()[0];

    xt::xtensor<float, 4> delta = p_delta + res_delta;;

    if (calc_delta_activation) {
        delta = delta * activation_derivative(outputs);
    }

    for (size_t b = 0; b < batch_size; b++) {
        for (size_t k = 0; k < num_filters; k++) {
            for (size_t i = 0; i < output_size[0]; i++) {
                for (size_t j = 0; j < output_size[1]; j++) {

                    float delta_val = delta(b, i, j, k);

                    // compute element-wise products for weights gradient propagation
                    for (size_t fi = 0; fi < filter_size; fi++) {
                        for (size_t fj = 0; fj < filter_size; fj++) {
                            size_t in_i = i * stride + fi;
                            size_t in_j = j * stride + fj;
                            for (size_t c = 0; c < input_size[2]; c++) {
                                grad_weights(k, fi, fj, c) += delta_val * input_activations(b, in_i, in_j, c);
                            }
                        }
                    }

                    grad_biases(k) += delta(b, i, j, k);
                }
            }
        }
    }

    xt::xtensor<float, 4> next_delta = xt::zeros<float>({batch_size, input_size[0], input_size[1], input_size[2]});

    for (size_t b = 0; b < batch_size; b++) {
        for (size_t k = 0; k < num_filters; k++) {
            for (size_t i = 0; i < output_size[0]; i++) {
                for (size_t j = 0; j < output_size[1]; j++) {

                    float delta_val = delta(b, i, j, k);

                    // back propagate delta values
                    for (size_t fi = 0; fi < filter_size; fi++) {
                        for (size_t fj = 0; fj < filter_size; fj++) {
                            size_t in_i = i * stride + fi;
                            size_t in_j = j * stride + fj;
                            for (size_t c = 0; c < input_size[2]; c++) {
                                next_delta(b, in_i, in_j, c) += delta_val * weights(k, fi, fj, c);
                            }
                        }
                    }
                }
            }
        }
    }

    grad_weights /= batch_size;
    grad_biases /= batch_size;

    return next_delta;
}

void Convolution::update(float lr) {
    weights -= lr * grad_weights;
    biases -= lr * grad_biases;

    grad_weights.fill(0);
    grad_biases.fill(0);
}

void Convolution::update_adam(float lr, float beta1, float beta2, size_t timestep) {
    update_adam_4d(weights, grad_weights, m_weights, v_weights, lr, beta1, beta2, timestep);
    update_adam_1d(biases, grad_biases, m_biases, v_biases, lr, beta1, beta2, timestep);
}

void Convolution::update_adamw(float lr, float beta1, float beta2, float weight_decay, size_t timestep) {
    update_adamw_4d(weights, grad_weights, m_weights, v_weights, lr, beta1, beta2, weight_decay, timestep);
    update_adam_1d(biases, grad_biases, m_biases, v_biases, lr, beta1, beta2, timestep);
}

void Convolution::save_weights(std::vector<float>& all) {
    save_4d(all, weights);
    save_4d(all, grad_weights);
    save_4d(all, m_weights);
    save_4d(all, v_weights);

    save_1d(all, biases);
    save_1d(all, grad_biases);
    save_1d(all, m_biases);
    save_1d(all, v_biases);
}

void Convolution::load_weights(xt::xtensor<float, 1>& all, size_t& index) {
    load_4d(all, weights, index);
    load_4d(all, grad_weights, index);
    load_4d(all, m_weights, index);
    load_4d(all, v_weights, index);

    load_1d(all, biases, index);
    load_1d(all, grad_biases, index);
    load_1d(all, m_biases, index);
    load_1d(all, v_biases, index);
}