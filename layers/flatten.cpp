
#include "flatten.h"


Flatten::Flatten(const std::vector<size_t>& p_input_size) {
    input_size = p_input_size;

    size_t product = 1;
    for (size_t e : input_size) product *= e;
    output_size = {product};
}


xt::xarray<float> Flatten::feedforward(const xt::xarray<float>& inputs, bool evaluation_mode) {
    batch_size = inputs.shape()[0];
    outputs = xt::reshape_view(inputs, std::vector<size_t>{batch_size, output_size[0]});
    return outputs;
}

xt::xarray<float> Flatten::backprop(const xt::xarray<float>& delta, bool calc_delta_activation) {
    auto real_shape = input_size;
    real_shape.insert(real_shape.begin(), batch_size);
    return xt::reshape_view(delta, real_shape);
}
