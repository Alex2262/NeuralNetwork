
#include "activation.h"
#include "../utilities.h"

Activation::Activation(const std::vector<size_t>& p_input_size, ActivationID p_activation_id) {
    input_size = p_input_size;
    output_size = input_size;

    activation_id = p_activation_id;

    activation_function = get_activation_function(activation_id);
    activation_derivative = get_activation_derivative(activation_id);
}


xt::xarray<float> Activation::feedforward(const xt::xarray<float>& inputs, bool evaluation_mode) {
    outputs = inputs;
    return activation_function(outputs);
}

xt::xarray<float> Activation::backprop(const xt::xarray<float>& p_delta, bool calc_delta_activation) {
    xt::xarray<float> delta = p_delta + res_delta;

    if (calc_delta_activation) {
        delta = delta * activation_derivative(outputs);
    }

    return delta;
}
