
#include "max_pool.h"
#include "../utilities.h"

MaxPool::MaxPool(std::vector<size_t>& p_input_size, size_t p_pool_size, size_t p_stride) {
    input_size = p_input_size;
    pool_size = p_pool_size;
    stride = p_stride;

    size_t out_h = (input_size[0] - p_pool_size) / stride + 1;
    size_t out_w = (input_size[1] - p_pool_size) / stride + 1;

    output_size = {out_h, out_w, input_size[2]};
}

xt::xarray<float> MaxPool::feedforward(const xt::xarray<float>& inputs, bool evaluation_mode) {
    size_t batch_size = inputs.shape()[0];
    input_activations = inputs;

    max_indices = xt::zeros<size_t>(std::vector<size_t>{batch_size, output_size[0], output_size[1], output_size[2], 2});
    outputs = xt::zeros<float>({batch_size, output_size[0], output_size[1], output_size[2]});

    for (size_t b = 0; b < batch_size; b++) {
        for (size_t i = 0; i < output_size[0]; i++) {
            for (size_t j = 0; j < output_size[1]; j++) {
                for (size_t c = 0; c < output_size[2]; c++) {

                    // Loop over pooling region and find max index
                    float max_val = -std::numeric_limits<float>::infinity();
                    size_t max_i = 0;
                    size_t max_j = 0;

                    for (size_t pi = 0; pi < pool_size; pi++) {
                        for (size_t pj = 0; pj < pool_size; pj++) {
                            size_t in_i = i * stride + pi;
                            size_t in_j = j * stride + pj;
                            float val = inputs(b,  in_i, in_j, c);
                            if (val > max_val) {
                                max_val = val;
                                max_i = pi;
                                max_j = pj;
                            }
                        }
                    }

                    outputs(b, i, j, c) = max_val;
                    max_indices(b, i, j, c, 0) = max_i;
                    max_indices(b, i, j, c, 1) = max_j;
                }
            }
        }
    }

    return outputs;
}

xt::xarray<float> MaxPool::backprop(const xt::xarray<float>& p_delta, bool calc_delta_activation) {
    size_t batch_size = input_activations.shape()[0];

    xt::xtensor<float, 4> delta = p_delta + res_delta;
    xt::xtensor<float, 4> next_delta = xt::zeros<float>({batch_size, input_size[0], input_size[1], input_size[2]});

    for (size_t b = 0; b < batch_size; b++) {
        for (size_t i = 0; i < output_size[0]; i++) {
            for (size_t j = 0; j < output_size[1]; j++) {
                for (size_t c = 0; c < output_size[2]; c++) {
                    next_delta(b, max_indices(b, i, j, c, 0), max_indices(b, i, j, c, 1), c) += delta(b, i, j, c);
                }
            }
        }
    }

    return next_delta;
}