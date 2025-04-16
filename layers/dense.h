
#ifndef NEURALNETWORK_DENSE_H
#define NEURALNETWORK_DENSE_H

#include "layers.h"
#include "../types.h"

class Dense : public Layer {
private:
    std::vector<size_t> input_size;
    std::vector<size_t> output_size;
    ActivationID activation_id;

    xt::xtensor<float, 2> weights, grad_weights, m_weights, v_weights;
    xt::xtensor<float, 1> biases, grad_biases, m_biases, v_biases;

    xt::xtensor<float, 2> input_activations;
    xt::xtensor<float, 2> activations;
    xt::xtensor<float, 2> outputs;

    ActivationFunction activation_function;
    ActivationDerivative activation_derivative;

public:
    Dense(const std::vector<size_t>& p_input_size, size_t p_num_neurons, ActivationID p_activation_id);

    xt::xarray<float> feedforward(const xt::xarray<float>& inputs, bool evaluation_mode) override;
    xt::xarray<float> backprop(const xt::xarray<float>& p_delta, bool calc_delta_activation) override;
    void update(float lr) override;
    void update_adam(float lr, float beta1, float beta2, float epsilon) override;

    [[nodiscard]] ActivationID get_activation_id() override { return activation_id; }
    [[nodiscard]] xt::xarray<float> get_outputs() override { return outputs; }
    [[nodiscard]] xt::xarray<float> get_activations() override { return activations; }
    [[nodiscard]] std::vector<size_t> get_input_size() const override { return input_size; }
    [[nodiscard]] std::vector<size_t> get_output_size() const override { return output_size; }
};


#endif //NEURALNETWORK_DENSE_H
