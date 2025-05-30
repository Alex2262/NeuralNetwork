//
// Created by Alexander Tian on 4/16/25.
//

#ifndef NEURALNETWORK_ACTIVATION_H
#define NEURALNETWORK_ACTIVATION_H

#include "layers.h"
#include "../types.h"


class Activation : public Layer {
private:
    std::vector<size_t> input_size;
    std::vector<size_t> output_size;

    ActivationID activation_id;
    ActivationFunction activation_function;
    ActivationDerivative activation_derivative;

    xt::xarray<float> outputs;
public:
    Activation(const std::vector<size_t>& p_input_size, ActivationID p_activation_id);

    xt::xarray<float> feedforward(const xt::xarray<float>& inputs, Mode mode) override;
    xt::xarray<float> backprop(const xt::xarray<float>& p_delta, bool calc_delta_activation) override;

    std::string get_name() const override { return "Activation"; }
    void update(float lr) override {};
    void update_adam(float lr, float beta1, float beta2, size_t timestep) override {};
    void update_adamw(float lr, float beta1, float beta2, float weight_decay, size_t timestep) override {};

    [[nodiscard]] ActivationID get_activation_id() override { return activation_id; }
    [[nodiscard]] xt::xarray<float> get_outputs() override { return outputs; }
    [[nodiscard]] xt::xarray<float> get_activations() override { return outputs; }
    [[nodiscard]] std::vector<size_t> get_input_size() const override { return input_size; }
    [[nodiscard]] std::vector<size_t> get_output_size() const override { return output_size; }
    [[nodiscard]] size_t get_num_params() const override { return 0; }

    void save_weights(std::vector<float>& all) override {};
    void load_weights(xt::xtensor<float, 1>& all, size_t& index) override {};
};


#endif //NEURALNETWORK_ACTIVATION_H
