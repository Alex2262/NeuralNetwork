
#ifndef NEURALNETWORK_CONVOLUTION_H
#define NEURALNETWORK_CONVOLUTION_H

#include "layers.h"
#include "../types.h"

class Convolution : public Layer {

private:
    std::vector<size_t> input_size;
    std::vector<size_t> output_size;

    size_t num_filters;
    size_t filter_size;
    size_t stride;

    size_t num_params;

    ActivationID activation_id;

    xt::xtensor<float, 4> weights, grad_weights, m_weights, v_weights;
    xt::xtensor<float, 1> biases, grad_biases, m_biases, v_biases;

    xt::xtensor<float, 4> input_activations;
    xt::xtensor<float, 4> activations;
    xt::xtensor<float, 4> outputs;

    ActivationFunction activation_function;
    ActivationDerivative activation_derivative;

public:
    Convolution(std::vector<size_t>& p_input_size, size_t p_num_filters, size_t p_filter_size, size_t p_stride,
                ActivationID p_activation_id);

    xt::xarray<float> feedforward(const xt::xarray<float>& inputs, Mode mode) override;
    xt::xarray<float> backprop(const xt::xarray<float>& p_delta, bool calc_delta_activation) override;

    std::string get_name() const override { return "Convolution"; }
    void update(float lr) override;
    void update_adam(float lr, float beta1, float beta2, size_t timestep) override;
    void update_adamw(float lr, float beta1, float beta2, float weight_decay, size_t timestep) override;

    [[nodiscard]] ActivationID get_activation_id() override { return activation_id; }
    [[nodiscard]] xt::xarray<float> get_outputs() override { return outputs; }
    [[nodiscard]] xt::xarray<float> get_activations() override { return activations; }
    [[nodiscard]] std::vector<size_t> get_input_size() const override { return input_size; }
    [[nodiscard]] std::vector<size_t> get_output_size() const override { return output_size; }
    [[nodiscard]] size_t get_num_params() const override { return num_params; }

    void save_weights(std::vector<float>& all) override;
    void load_weights(xt::xtensor<float, 1>& all, size_t& index) override;

    void zero_grad() override {
        grad_weights.fill(0);
        grad_biases.fill(0);
        res_delta = 0;
    }
};


#endif //NEURALNETWORK_CONVOLUTION_H
