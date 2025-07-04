
#ifndef NEURALNETWORK_EMBEDDING_H
#define NEURALNETWORK_EMBEDDING_H

#include "layers.h"
#include "../types.h"

class Embedding : public Layer {
private:
    size_t vocab_size;
    size_t d_model;
    size_t max_seq_len;

    size_t num_params;

    std::vector<size_t> input_size;
    std::vector<size_t> output_size;

    xt::xtensor<float, 2> embedding_matrix, grad_embedding_matrix, m_embedding_matrix, v_embedding_matrix;
    xt::xtensor<float, 2> positional_matrix, grad_positional_matrix, m_positional_matrix, v_positional_matrix;

    xt::xtensor<float, 2> input_activation;  // {batch_size, seq_len}
    xt::xtensor<float, 2> outputs;  // {batch_size * seq_len, d_model}

public:
    Embedding(const std::vector<size_t>& p_input_size, size_t p_vocab_size, size_t p_d_model, ActivationID p_activation_id);

    xt::xarray<float> feedforward(const xt::xarray<float>& inputs, Mode mode) override;
    xt::xarray<float> backprop(const xt::xarray<float>& p_delta, bool calc_delta_activation) override;

    std::string get_name() const override { return "Embedding"; }
    void update(float lr) override;
    void update_adam(float lr, float beta1, float beta2, size_t timestep) override;
    void update_adamw(float lr, float beta1, float beta2, float weight_decay, size_t timestep) override;

    [[nodiscard]] ActivationID get_activation_id() override { return ActivationID::NONE; }
    [[nodiscard]] xt::xarray<float> get_outputs() override { return outputs; }
    [[nodiscard]] xt::xarray<float> get_activations() override { return outputs; }
    [[nodiscard]] std::vector<size_t> get_input_size() const override { return input_size; }
    [[nodiscard]] std::vector<size_t> get_output_size() const override { return output_size; }
    [[nodiscard]] size_t get_num_params() const override { return num_params; }

    // specific to embedding layer, to support projection layer
    [[nodiscard]] xt::xtensor<float, 2>& get_embedding_matrix() { return embedding_matrix; }
    [[nodiscard]] xt::xtensor<float, 2>& get_grad_embedding_matrix() { return grad_embedding_matrix; }

    void save_weights(std::vector<float>& all) override;
    void load_weights(xt::xtensor<float, 1>& all, size_t& index) override;

    void zero_grad() override {
        grad_embedding_matrix.fill(0);
        grad_positional_matrix.fill(0);
        res_delta = 0;
    }
};

#endif //NEURALNETWORK_EMBEDDING_H
