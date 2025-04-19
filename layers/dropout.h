

#ifndef NEURALNETWORK_DROPOUT_H
#define NEURALNETWORK_DROPOUT_H


#include "layers.h"
#include "../types.h"


class Dropout : public Layer {
private:
    std::vector<size_t> input_size;
    std::vector<size_t> output_size;

    float keep_rate;
    xt::xarray<float> outputs;
    xt::xarray<float> mask;

public:
    Dropout(const std::vector<size_t>& p_input_size, float p_dropout_rate);

    xt::xarray<float> feedforward(const xt::xarray<float>& inputs, bool evaluation_mode) override;
    xt::xarray<float> backprop(const xt::xarray<float>& p_delta, bool calc_delta_activation) override;

    std::string get_name() const override { return "Dropout"; }
    void update(float lr) override {};
    void update_adam(float lr, float beta1, float beta2, float epsilon) override {};

    [[nodiscard]] ActivationID get_activation_id() override { return ActivationID::NONE; }
    [[nodiscard]] xt::xarray<float> get_outputs() override { return outputs; }
    [[nodiscard]] xt::xarray<float> get_activations() override { return outputs; }
    [[nodiscard]] std::vector<size_t> get_input_size() const override { return input_size; }
    [[nodiscard]] std::vector<size_t> get_output_size() const override { return output_size; }
};

#endif //NEURALNETWORK_DROPOUT_H
