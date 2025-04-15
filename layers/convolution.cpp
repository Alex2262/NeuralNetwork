
#include <xtensor/xrandom.hpp>
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

    weights = xt::random::randn<float>({num_filters, filter_size, filter_size, input_size[2]});
    biases = xt::random::randn<float>({num_filters});

    grad_weights = xt::zeros<float>({num_filters, filter_size, filter_size, input_size[2]});
    grad_biases = xt::zeros<float>({num_filters});

    activation_id = p_activation_id;

    activation_function = get_activation_function(activation_id);
    activation_derivative = get_activation_derivative(activation_id);
}


xt::xarray<float> Convolution::feedforward(const xt::xarray<float>& inputs, bool evaluation_mode) {
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

xt::xarray<float> Convolution::backprop(const xt::xarray<float>& delta, bool calc_delta_activation) {
    size_t batch_size = input_activations.shape()[0];

    xt::xtensor<float, 4> prop_delta = delta;

    if (calc_delta_activation) {
        prop_delta = delta * activation_derivative(outputs);
    }

    for (size_t b = 0; b < batch_size; b++) {
        for (size_t k = 0; k < num_filters; k++) {
            for (size_t i = 0; i < output_size[0]; i++) {
                for (size_t j = 0; j < output_size[1]; j++) {

                    float delta_val = prop_delta(b, i, j, k);

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

                    grad_biases(k) += prop_delta(b, i, j, k);
                }
            }
        }
    }

    xt::xtensor<float, 4> next_delta = xt::zeros<float>({batch_size, input_size[0], input_size[1], input_size[2]});

    for (size_t b = 0; b < batch_size; b++) {
        for (size_t k = 0; k < num_filters; k++) {
            for (size_t i = 0; i < output_size[0]; i++) {
                for (size_t j = 0; j < output_size[1]; j++) {

                    float delta_val = prop_delta(b, i, j, k);

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

    return next_delta;
}

void Convolution::update(float lr) {
    weights -= lr * grad_weights;
    biases -= lr * grad_biases;

    grad_weights.fill(0);
    grad_biases.fill(0);
}
