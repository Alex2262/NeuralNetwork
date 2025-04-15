
#ifndef NEURALNETWORK_FLATTEN_H
#define NEURALNETWORK_FLATTEN_H

#include "layers.h"
#include "../types.h"


class Flatten : public Layer {
private:
    std::vector<size_t> input_size;
    std::vector<size_t> output_size;

    size_t batch_size;
    xt::xtensor<float, 2> outputs;

public:
    explicit Flatten(const std::vector<size_t>& p_input_size);

    xt::xarray<float> feedforward(const xt::xarray<float>& inputs, bool evaluation_mode) override;
    xt::xarray<float> backprop(const xt::xarray<float>& delta, bool calc_delta_activation) override;

    void update(float lr) override {};

    [[nodiscard]] ActivationID get_activation_id() override { return ActivationID::NONE; }
    [[nodiscard]] xt::xarray<float> get_outputs() override { return outputs; }
    [[nodiscard]] std::vector<size_t> get_input_size() const override { return input_size; }
    [[nodiscard]] std::vector<size_t> get_output_size() const override { return output_size; }
};

#endif //NEURALNETWORK_FLATTEN_H
