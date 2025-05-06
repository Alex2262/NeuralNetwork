
#ifndef NEURALNETWORK_ATTENTION_H
#define NEURALNETWORK_ATTENTION_H


#include "layers.h"
#include "../types.h"

// Masked Multi Head Attention Layer
class Attention : public Layer {
private:
    size_t num_heads;
    size_t d_model;
    size_t d_k;
    size_t max_seq_len;

    size_t num_params;

    std::vector<size_t> input_size;
    std::vector<size_t> output_size;

    xt::xtensor<float, 2> weights_q, grad_weights_q, m_weights_q, v_weights_q;
    xt::xtensor<float, 2> weights_k, grad_weights_k, m_weights_k, v_weights_k;
    xt::xtensor<float, 2> weights_v, grad_weights_v, m_weights_v, v_weights_v;
    xt::xtensor<float, 2> weights_o, grad_weights_o, m_weights_o, v_weights_o;

    std::vector<xt::xtensor<float, 2>> Qi;
    std::vector<xt::xtensor<float, 2>> Ki;
    std::vector<xt::xtensor<float, 2>> Vi;
    std::vector<xt::xtensor<float, 2>> Ri;
    std::vector<xt::xtensor<float, 2>> Ai;

    xt::xtensor<float, 2> E;
    xt::xtensor<float, 2> A;
    xt::xtensor<float, 3> C;

    xt::xtensor<float, 2> C_reshaped;

    xt::xtensor<float, 2> outputs;  // {batch_size * seq_len, d_model}

public:
    Attention(const std::vector<size_t>& p_input_size, size_t p_num_heads, ActivationID p_activation_id);

    xt::xarray<float> feedforward(const xt::xarray<float>& inputs, bool evaluation_mode) override;
    xt::xarray<float> backprop(const xt::xarray<float>& p_delta, bool calc_delta_activation) override;

    std::string get_name() const override { return "Attention"; }
    void update(float lr) override;
    void update_adam(float lr, float beta1, float beta2, float epsilon) override;
    void update_adamw(float lr, float beta1, float beta2, float epsilon, float weight_decay) override;

    [[nodiscard]] ActivationID get_activation_id() override { return ActivationID::NONE; }
    [[nodiscard]] xt::xarray<float> get_outputs() override { return outputs; }
    [[nodiscard]] xt::xarray<float> get_activations() override { return outputs; }
    [[nodiscard]] std::vector<size_t> get_input_size() const override { return input_size; }
    [[nodiscard]] std::vector<size_t> get_output_size() const override { return output_size; }
    [[nodiscard]] size_t get_num_params() const override { return num_params; }
};


#endif //NEURALNETWORK_ATTENTION_H
