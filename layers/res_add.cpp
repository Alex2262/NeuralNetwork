
#include "res_add.h"



ResAdd::ResAdd(const std::vector<size_t>& p_input_size, Layer* p_res_layer) {
    input_size = p_input_size;
    output_size = input_size;

    res_layer = p_res_layer;
}

xt::xarray<float> ResAdd::feedforward(const xt::xarray<float>& inputs, bool evaluation_mode) {
    const xt::xarray<float>& res_inputs = res_layer->get_activations();

    assert(std::equal(inputs.shape().begin(), inputs.shape().end(),
                      res_inputs.shape().begin(), res_inputs.shape().end()));

    outputs = res_inputs + inputs;

    return outputs;
}


xt::xarray<float> ResAdd::backprop(const xt::xarray<float>& p_delta, bool calc_delta_activation) {
    xt::xarray<float> delta = p_delta + res_delta;
    res_layer->set_res_delta(delta);
    return delta;
}