
#include <xtensor/xrandom.hpp>
#include <xtensor-blas/xlinalg.hpp>

#include "projection.h"
#include "../utilities.h"

Projection::Projection(const std::vector<size_t>& p_input_size, Embedding* p_embedding_layer, ActivationID p_activation_id) {
    embedding_layer = p_embedding_layer;
    xt::xtensor<float, 2>& embedding_matrix = embedding_layer->get_embedding_matrix();

    vocab_size = embedding_matrix.shape()[0];
    d_model = embedding_matrix.shape()[1];
    max_seq_len = p_input_size[0];

    input_size = p_input_size;
    output_size = {max_seq_len, vocab_size};

    activation_id = p_activation_id;
    activation_function = get_activation_function(activation_id);
    activation_derivative = get_activation_derivative(activation_id);

    timestep = 0;
}

xt::xarray<float> Projection::feedforward(const xt::xarray<float>& inputs, bool evaluation_mode) {
    input_activation = inputs;

    size_t seq_len = max_seq_len;
    size_t batch_size = inputs.shape()[0] / seq_len;

    xt::xtensor<float, 2>& embedding_matrix = embedding_layer->get_embedding_matrix();
    xt::xtensor<float, 2> raw_outputs = xt::linalg::dot(inputs, xt::transpose(embedding_matrix));

    outputs = xt::reshape_view(raw_outputs, {batch_size, seq_len, vocab_size});
    activations = xt::reshape_view(activation_function(raw_outputs), {batch_size, seq_len, vocab_size});

    return activations;
}

xt::xarray<float> Projection::backprop(const xt::xarray<float>& p_delta, bool calc_delta_activation) {
    assert(!calc_delta_activation);

    size_t seq_len = max_seq_len;
    size_t batch_size = input_activation.shape()[0] / seq_len;

    xt::xtensor<float, 2> delta = xt::reshape_view(xt::eval(p_delta + res_delta), {batch_size * seq_len, vocab_size});

    // note this is unnecessary for the projection layer since it's the last layer.
    // xt::xtensor<float, 2> next_delta = delta;
    // if (calc_delta_activation) {
    //     next_delta = delta * activation_derivative(outputs);
    // }

    xt::xtensor<float, 2>& embedding_matrix = embedding_layer->get_embedding_matrix();
    xt::xtensor<float, 2>& grad_embedding_matrix = embedding_layer->get_grad_embedding_matrix();

    grad_embedding_matrix += xt::linalg::dot(xt::transpose(delta), input_activation);

    xt::xtensor<float, 2> delta_out = xt::linalg::dot(delta, embedding_matrix);
    return delta_out;
}