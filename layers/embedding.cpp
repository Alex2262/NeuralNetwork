
#include <xtensor/generators/xrandom.hpp>
#include <xtensor-blas/xlinalg.hpp>

#include "embedding.h"
#include "../utilities.h"

Embedding::Embedding(const std::vector<size_t>& p_input_size, size_t p_vocab_size, size_t p_d_model, ActivationID p_activation_id) {
    vocab_size = p_vocab_size;
    d_model = p_d_model;
    max_seq_len = p_input_size[0];

    input_size = p_input_size;
    output_size = {max_seq_len, d_model};

    float stddev = 1.0f / std::sqrt(static_cast<float>(d_model));
    embedding_matrix = stddev * xt::random::rand<float>({vocab_size, d_model});
    positional_matrix = stddev * xt::random::rand<float>({max_seq_len, d_model});

    grad_embedding_matrix = xt::zeros_like(embedding_matrix);
    grad_positional_matrix = xt::zeros_like(positional_matrix);

    m_embedding_matrix = xt::zeros_like(embedding_matrix);
    m_positional_matrix = xt::zeros_like(positional_matrix);

    v_embedding_matrix = xt::zeros_like(embedding_matrix);
    v_positional_matrix = xt::zeros_like(positional_matrix);

    num_params = vocab_size * d_model + max_seq_len * d_model;
}

xt::xarray<float> Embedding::feedforward(const xt::xarray<float>& inputs, Mode mode) {
    input_activation = inputs;
    size_t batch_size = input_activation.shape()[0];
    size_t seq_len = input_activation.shape()[1];

    xt::xtensor<float, 3> token_embeds = xt::zeros<float>({batch_size, seq_len, d_model});
    xt::xtensor<float, 3> position_embeds = xt::zeros<float>({batch_size, seq_len, d_model});

    for (size_t b = 0; b < batch_size; b++) {
        for (size_t i = 0; i < seq_len; i++) {
            auto token_idx = static_cast<size_t>(input_activation(b, i));

            for (size_t d = 0; d < d_model; d++) {
                token_embeds(b, i, d) = embedding_matrix(token_idx, d);
                position_embeds(b, i, d) = positional_matrix(i, d);
            }
        }
    }

    outputs = xt::reshape_view(xt::eval(token_embeds + position_embeds), {batch_size * seq_len, d_model});
    return outputs;
}

xt::xarray<float> Embedding::backprop(const xt::xarray<float>& p_delta, bool calc_delta_activation) {
    size_t batch_size = input_activation.shape()[0];
    size_t seq_len = input_activation.shape()[1];
    xt::xtensor<float, 3> delta = xt::reshape_view(xt::eval(p_delta + res_delta), {batch_size, seq_len, d_model});

    for (size_t b = 0; b < batch_size; b++) {
        for (size_t i = 0; i < seq_len; i++) {
            auto token_idx = static_cast<size_t>(input_activation(b, i));

            for (size_t d = 0; d < d_model; d++) {
                float delta_val = delta(b, i, d);
                grad_embedding_matrix(token_idx, d) += delta_val;
                grad_positional_matrix(i, d) += delta_val;
            }
        }
    }

    auto div = static_cast<float>(batch_size * seq_len);
    grad_embedding_matrix /= div;
    grad_positional_matrix /= div;

    // we return an empty xarray here because we do not require any more backpropagation.
    return xt::zeros<float>({batch_size});
}

void Embedding::update(float lr) {
    embedding_matrix -= lr * grad_embedding_matrix;
    positional_matrix -= lr * grad_positional_matrix;

    grad_embedding_matrix.fill(0);
    grad_positional_matrix.fill(0);
}

void Embedding::update_adam(float lr, float beta1, float beta2, size_t timestep) {
    update_adam_2d(embedding_matrix, grad_embedding_matrix, m_embedding_matrix, v_embedding_matrix, lr, beta1, beta2, timestep);
    update_adam_2d(positional_matrix, grad_positional_matrix, m_positional_matrix, v_positional_matrix, lr, beta1, beta2, timestep);
}

void Embedding::update_adamw(float lr, float beta1, float beta2, float weight_decay, size_t timestep) {
    update_adam(lr, beta1, beta2, timestep);
}


void Embedding::save_weights(std::vector<float>& all) {
    save_2d(all, embedding_matrix);
    save_2d(all, grad_embedding_matrix);
    save_2d(all, m_embedding_matrix);
    save_2d(all, v_embedding_matrix);

    save_2d(all, positional_matrix);
    save_2d(all, grad_positional_matrix);
    save_2d(all, m_positional_matrix);
    save_2d(all, v_positional_matrix);
}

void Embedding::load_weights(xt::xtensor<float, 1>& all, size_t& index) {
    load_2d(all, embedding_matrix, index);
    load_2d(all, grad_embedding_matrix, index);
    load_2d(all, m_embedding_matrix, index);
    load_2d(all, v_embedding_matrix, index);

    load_2d(all, positional_matrix, index);
    load_2d(all, grad_positional_matrix, index);
    load_2d(all, m_positional_matrix, index);
    load_2d(all, v_positional_matrix, index);
}