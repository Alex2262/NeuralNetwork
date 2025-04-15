//
// Created by Alexander Tian on 3/19/25.
//

#ifndef NEURALNETWORK_MAX_POOL_H
#define NEURALNETWORK_MAX_POOL_H


#include "layers.h"
#include "../types.h"

class MaxPool : public Layer {

private:
    std::vector<size_t> input_size;
    std::vector<size_t> output_size;

    size_t pool_size;
    size_t stride;

    xt::xtensor<float, 4> input_activations;
    xt::xtensor<float, 4> outputs;
    xt::xtensor<size_t, 5> max_indices;

public:
    MaxPool(std::vector<size_t>& p_input_size, size_t p_pool_size, size_t p_stride);

    xt::xarray<float> feedforward(const xt::xarray<float>& inputs, bool evaluation_mode) override;
    xt::xarray<float> backprop(const xt::xarray<float>& delta, bool calc_delta_activation) override;

    void update(float lr) override {};
    void update_adam(float lr, float beta1, float beta2, float epsilon) override {};

    [[nodiscard]] ActivationID get_activation_id() override { return ActivationID::NONE; }
    [[nodiscard]] xt::xarray<float> get_outputs() override { return outputs; }
    [[nodiscard]] std::vector<size_t> get_input_size() const override { return input_size; }
    [[nodiscard]] std::vector<size_t> get_output_size() const override { return output_size; }
};


#endif //NEURALNETWORK_MAX_POOL_H
