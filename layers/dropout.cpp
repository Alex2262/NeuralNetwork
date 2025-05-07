
#include <xtensor/generators/xrandom.hpp>
#include <xtensor-blas/xlinalg.hpp>

#include "dropout.h"

Dropout::Dropout(const std::vector<size_t>& p_input_size, float p_dropout_rate) {
    input_size = p_input_size;
    output_size = input_size;

    keep_rate = 1.0f - p_dropout_rate;
}


xt::xarray<float> Dropout::feedforward(const xt::xarray<float>& inputs, Mode mode) {
    if (mode != Mode::TRAINING) {
        mask = xt::ones_like(inputs);
        outputs = inputs;
        return outputs;
    }

    xt::xarray<float> random = xt::random::rand<float>(inputs.shape(), 0.0f, 1.0f);
    mask = xt::cast<float>(random < keep_rate);
    mask /= keep_rate;

    outputs = inputs * mask;
    return outputs;
}

xt::xarray<float> Dropout::backprop(const xt::xarray<float>& p_delta, bool calc_delta_activation) {
    xt::xarray<float> delta = p_delta + res_delta;
    return delta * mask;
}