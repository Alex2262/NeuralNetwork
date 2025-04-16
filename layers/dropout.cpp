
#include <xtensor/xrandom.hpp>
#include <xtensor-blas/xlinalg.hpp>

#include "dropout.h"

Dropout::Dropout(const std::vector<size_t>& p_input_size, float p_dropout_rate) {
    input_size = p_input_size;
    output_size = input_size;

    keep_rate = 1.0f - p_dropout_rate;
}


xt::xarray<float> Dropout::feedforward(const xt::xarray<float>& inputs, bool evaluation_mode) {
    if (evaluation_mode) {
        mask = xt::ones_like(inputs);
        return inputs;
    }

    xt::xarray<float> random = xt::random::rand<float>(inputs.shape(), 0.0f, 1.0f);
    mask = xt::cast<float>(random < keep_rate);
    mask /= keep_rate;

    outputs = inputs * mask;
    return outputs;
}

xt::xarray<float> Dropout::backprop(const xt::xarray<float>& p_delta, bool calc_delta_activation) {
    return p_delta * mask;
}