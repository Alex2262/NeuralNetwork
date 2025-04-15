
#include <xtensor/xrandom.hpp>
#include <xtensor-blas/xlinalg.hpp>

#include "attention.h"
#include "../utilities.h"

Attention::Attention(const std::vector<size_t>& p_input_size, size_t p_num_heads, ActivationID p_activation_id) {
    input_size = p_input_size;
    output_size = input_size;

    num_heads = p_num_heads;
    max_seq_len = input_size[0];
    d_model = input_size[1];

    // sanity check for assigning a value to d_k
    if (d_model % num_heads != 0) {
        throw std::runtime_error("Error: d_model must be divisible by num_heads.");
    }

    d_k = d_model / num_heads;

    weights_q = xt::random::rand<float>({d_model, d_model}, -1.0, 1.0);
    weights_k = xt::random::rand<float>({d_model, d_model}, -1.0, 1.0);
    weights_v = xt::random::rand<float>({d_model, d_model}, -1.0, 1.0);
    weights_o = xt::random::rand<float>({d_model, d_model}, -1.0, 1.0);

    grad_weights_q = xt::zeros<float>({d_model, d_model});
    grad_weights_k = xt::zeros<float>({d_model, d_model});
    grad_weights_v = xt::zeros<float>({d_model, d_model});
    grad_weights_o = xt::zeros<float>({d_model, d_model});
}

xt::xarray<float> Attention::feedforward(const xt::xarray<float>& inputs, bool evaluation_mode) {
    input_activations = inputs;

    size_t batch_size = input_activations.shape()[0];
    size_t seq_len = inputs.shape()[1];

    E = xt::reshape_view(inputs, {batch_size * seq_len, d_model});

    xt::xtensor<float, 3> Q = xt::reshape_view(xt::linalg::dot(E, weights_q), {batch_size * num_heads, seq_len, d_k});
    xt::xtensor<float, 3> K = xt::reshape_view(xt::linalg::dot(E, weights_k), {batch_size * num_heads, seq_len, d_k});
    xt::xtensor<float, 3> V = xt::reshape_view(xt::linalg::dot(E, weights_v), {batch_size * num_heads, seq_len, d_k});


    xt::xtensor<float, 2> mask = xt::zeros<float>({seq_len, seq_len});

    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = i + 1; j < seq_len; j++) {
            mask(i, j) = -std::numeric_limits<float>::infinity();
        }
    }

    C = xt::zeros<float>({batch_size * num_heads, seq_len, d_k});

    Qi.resize(batch_size * num_heads);
    Ki.resize(batch_size * num_heads);
    Vi.resize(batch_size * num_heads);
    Ri.resize(batch_size * num_heads);
    Ai.resize(batch_size * num_heads);

    for (size_t batch = 0; batch < batch_size; batch++) {
        for (size_t head = 0; head < num_heads; head++) {
            size_t index = batch * num_heads + head;

            Qi[index] = index_3d(Q, index);
            Ki[index] = index_3d(K, index);
            Vi[index] = index_3d(V, index);

            Ri[index] = xt::linalg::dot(Qi[index], xt::transpose(Ki[index])) / sqrt(d_k) + mask;
            Ai[index] = softmax(Ri[index]);

            xt::xtensor<float, 2> Ci = xt::linalg::dot(Ai[index], Vi[index]);
            set_3d(C, Ci, index);
        }
    }

    C_reshaped = xt::reshape_view(C, {batch_size * seq_len, num_heads * d_k});

    xt::xtensor<float, 2> O = xt::linalg::dot(C_reshaped, weights_o);

    outputs = E + O;
    return outputs;
}

xt::xarray<float> Attention::backprop(const xt::xarray<float>& delta, bool calc_delta_activation) {
    size_t batch_size = input_activations.shape()[0];
    size_t seq_len = input_activations.shape()[1];

    xt::xtensor<float, 2> delta_O = delta;
    xt::xtensor<float, 2> delta_C = xt::linalg::dot(delta_O, xt::transpose(weights_o));
    xt::xtensor<float, 3> delta_C_reshaped = xt::reshape_view(delta_C, {batch_size * num_heads, seq_len, d_k});

    grad_weights_o += xt::linalg::dot(xt::transpose(C_reshaped), delta_O);

    xt::xtensor<float, 3> delta_Q = xt::zeros<float>({batch_size * num_heads, seq_len, d_k});
    xt::xtensor<float, 3> delta_K = xt::zeros<float>({batch_size * num_heads, seq_len, d_k});
    xt::xtensor<float, 3> delta_V = xt::zeros<float>({batch_size * num_heads, seq_len, d_k});

    for (size_t batch = 0; batch < batch_size; batch++) {
        for (size_t head = 0; head < num_heads; head++) {
            size_t index = batch * num_heads + head;

            xt::xtensor<float, 2> delta_Ci = index_3d(delta_C_reshaped, index);
            xt::xtensor<float, 2> delta_Vi = xt::linalg::dot(xt::transpose(Ai[index]), delta_Ci);
            xt::xtensor<float, 2> delta_Ai = xt::linalg::dot(delta_Ci, xt::transpose(Vi[index]));

            // Delta_Ri calculation with Softmax Jacobian
            xt::xtensor<float, 2> prod = Ai[index] * delta_Ai;
            xt::xtensor<float, 1> row_sum = xt::sum(prod, {1});
            xt::xtensor<float, 2> row_sum_expanded = xt::expand_dims(row_sum, 1);
            xt::xtensor<float, 2> delta_Ri = Ai[index] * (delta_Ai - row_sum_expanded);

            xt::xtensor<float, 2> delta_Qi = xt::linalg::dot(delta_Ri, Ki[index]) / sqrt(d_k);
            xt::xtensor<float, 2> delta_Ki = xt::linalg::dot(xt::transpose(delta_Ri), Qi[index]) / sqrt(d_k);

            set_3d(delta_Q, delta_Qi, index);
            set_3d(delta_K, delta_Ki, index);
            set_3d(delta_V, delta_Vi, index);
        }
    }

    xt::xtensor<float, 2> E_T = xt::transpose(E);
    grad_weights_q += xt::linalg::dot(E_T, delta_Q);
    grad_weights_k += xt::linalg::dot(E_T, delta_K);
    grad_weights_v += xt::linalg::dot(E_T, delta_V);

    xt::xtensor<float, 2> delta_E = xt::linalg::dot(delta_Q, xt::transpose(weights_q)) +
                                    xt::linalg::dot(delta_K, xt::transpose(weights_k)) +
                                    xt::linalg::dot(delta_V, xt::transpose(weights_v)) +
                                    delta;

    return xt::reshape_view(delta_E, {batch_size, seq_len, d_model});
}

void Attention::update(float lr) {
    weights_q -= lr * grad_weights_q;
    weights_k -= lr * grad_weights_k;
    weights_v -= lr * grad_weights_v;
    weights_o -= lr * grad_weights_o;

    grad_weights_q.fill(0);
    grad_weights_k.fill(0);
    grad_weights_v.fill(0);
    grad_weights_o.fill(0);
}
