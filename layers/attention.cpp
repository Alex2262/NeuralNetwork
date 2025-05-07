
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

    float stddev = std::sqrt(1.0f / static_cast<float>(d_model));
    weights_q = stddev * xt::random::randn<float>({d_model, d_model});
    weights_k = stddev * xt::random::randn<float>({d_model, d_model});
    weights_v = stddev * xt::random::randn<float>({d_model, d_model});
    weights_o = stddev * xt::random::randn<float>({d_model, d_model});

    grad_weights_q = xt::zeros_like(weights_q);
    grad_weights_k = xt::zeros_like(weights_k);
    grad_weights_v = xt::zeros_like(weights_v);
    grad_weights_o = xt::zeros_like(weights_o);

    m_weights_q = xt::zeros_like(weights_q);
    m_weights_k = xt::zeros_like(weights_k);
    m_weights_v = xt::zeros_like(weights_v);
    m_weights_o = xt::zeros_like(weights_o);

    v_weights_q = xt::zeros_like(weights_q);
    v_weights_k = xt::zeros_like(weights_k);
    v_weights_v = xt::zeros_like(weights_v);
    v_weights_o = xt::zeros_like(weights_o);

    timestep = 0;

    num_params = 4 * d_model * d_model;

    mask = xt::zeros<float>({max_seq_len, max_seq_len});

    for (size_t i = 0; i < max_seq_len; i++) {
        for (size_t j = i + 1; j < max_seq_len; j++) {
            mask(i, j) = -1e9f;
        }
    }
}

xt::xarray<float> Attention::feedforward(const xt::xarray<float>& inputs, Mode mode) {
    size_t seq_len = max_seq_len;
    size_t batch_size = inputs.shape()[0] / seq_len;

    E = inputs;

    xt::xtensor<float, 3> Q = xt::eval(xt::reshape_view(xt::linalg::dot(E, weights_q), {batch_size * num_heads, seq_len, d_k}));
    xt::xtensor<float, 3> K = xt::eval(xt::reshape_view(xt::linalg::dot(E, weights_k), {batch_size * num_heads, seq_len, d_k}));
    xt::xtensor<float, 3> V = xt::eval(xt::reshape_view(xt::linalg::dot(E, weights_v), {batch_size * num_heads, seq_len, d_k}));

    C = xt::zeros<float>({batch_size * num_heads, seq_len, d_k});

    if (Qi.size() != batch_size * num_heads) {
        Qi.resize(batch_size * num_heads);
        Ki.resize(batch_size * num_heads);
        Vi.resize(batch_size * num_heads);
        Ri.resize(batch_size * num_heads);
        Ai.resize(batch_size * num_heads);
    }

    for (size_t batch = 0; batch < batch_size; batch++) {
        for (size_t head = 0; head < num_heads; head++) {
            size_t index = batch * num_heads + head;

            Qi[index] = index_3d(Q, index);
            Ki[index] = index_3d(K, index);
            Vi[index] = index_3d(V, index);

            Ri[index] = xt::linalg::dot(Qi[index], xt::eval(xt::transpose(Ki[index]))) / sqrt(d_k) + mask;
            Ai[index] = softmax(Ri[index]);

            xt::xtensor<float, 2> Ci = xt::linalg::dot(Ai[index], Vi[index]);
            set_3d(C, Ci, index);
        }
    }

    C_reshaped = xt::eval(xt::reshape_view(C, {batch_size * seq_len, num_heads * d_k}));

    outputs = xt::linalg::dot(C_reshaped, weights_o);

    return outputs;
}

xt::xarray<float> Attention::backprop(const xt::xarray<float>& p_delta, bool calc_delta_activation) {
    xt::xtensor<float, 2> delta = p_delta + res_delta;

    size_t seq_len = max_seq_len;
    size_t batch_size = E.shape()[0] / seq_len;

    xt::xtensor<float, 2> delta_C = xt::linalg::dot(delta, xt::eval(xt::transpose(weights_o)));
    xt::xtensor<float, 3> delta_C_reshaped = xt::eval(xt::reshape_view(delta_C, {batch_size * num_heads, seq_len, d_k}));

    grad_weights_o += xt::linalg::dot(xt::eval(xt::transpose(C_reshaped)), delta);

    xt::xtensor<float, 3> delta_Q = xt::zeros<float>({batch_size * num_heads, seq_len, d_k});
    xt::xtensor<float, 3> delta_K = xt::zeros<float>({batch_size * num_heads, seq_len, d_k});
    xt::xtensor<float, 3> delta_V = xt::zeros<float>({batch_size * num_heads, seq_len, d_k});

    for (size_t batch = 0; batch < batch_size; batch++) {
        for (size_t head = 0; head < num_heads; head++) {
            size_t index = batch * num_heads + head;

            xt::xtensor<float, 2> delta_Ci = index_3d(delta_C_reshaped, index);
            xt::xtensor<float, 2> delta_Vi = xt::linalg::dot(xt::eval(xt::transpose(Ai[index])), delta_Ci);
            xt::xtensor<float, 2> delta_Ai = xt::linalg::dot(delta_Ci, xt::eval(xt::transpose(Vi[index])));

            // Delta_Ri calculation with Softmax Jacobian
            xt::xtensor<float, 2> prod = Ai[index] * delta_Ai;
            xt::xtensor<float, 1> row_sum = xt::sum(prod, {1});
            xt::xtensor<float, 2> row_sum_expanded = xt::expand_dims(row_sum, 1);
            xt::xtensor<float, 2> delta_Ri = Ai[index] * (delta_Ai - row_sum_expanded);

            xt::xtensor<float, 2> delta_Qi = xt::linalg::dot(delta_Ri, Ki[index]) / sqrt(d_k);
            xt::xtensor<float, 2> delta_Ki = xt::linalg::dot(xt::eval(xt::transpose(delta_Ri)), Qi[index]) / sqrt(d_k);

            set_3d(delta_Q, delta_Qi, index);
            set_3d(delta_K, delta_Ki, index);
            set_3d(delta_V, delta_Vi, index);
        }
    }

    xt::xtensor<float, 2> delta_Q_reshaped = xt::eval(xt::reshape_view(delta_Q, {batch_size * seq_len, num_heads * d_k}));
    xt::xtensor<float, 2> delta_K_reshaped = xt::eval(xt::reshape_view(delta_K, {batch_size * seq_len, num_heads * d_k}));
    xt::xtensor<float, 2> delta_V_reshaped = xt::eval(xt::reshape_view(delta_V, {batch_size * seq_len, num_heads * d_k}));

    xt::xtensor<float, 2> E_T = xt::eval(xt::transpose(E));
    grad_weights_q += xt::linalg::dot(E_T, delta_Q_reshaped);
    grad_weights_k += xt::linalg::dot(E_T, delta_K_reshaped);
    grad_weights_v += xt::linalg::dot(E_T, delta_V_reshaped);

    xt::xtensor<float, 2> delta_E = xt::linalg::dot(delta_Q_reshaped, xt::eval(xt::transpose(weights_q))) +
                                    xt::linalg::dot(delta_K_reshaped, xt::eval(xt::transpose(weights_k))) +
                                    xt::linalg::dot(delta_V_reshaped, xt::eval(xt::transpose(weights_v)));

    auto div = static_cast<float>(batch_size * seq_len);
    grad_weights_q /= div;
    grad_weights_k /= div;
    grad_weights_v /= div;
    grad_weights_o /= div;

    return delta_E;
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

void Attention::update_adam(float lr, float beta1, float beta2) {
    timestep++;

    update_adam_2d(weights_q, grad_weights_q, m_weights_q, v_weights_q, lr, beta1, beta2, timestep);
    update_adam_2d(weights_k, grad_weights_k, m_weights_k, v_weights_k, lr, beta1, beta2, timestep);
    update_adam_2d(weights_v, grad_weights_v, m_weights_v, v_weights_v, lr, beta1, beta2, timestep);
    update_adam_2d(weights_o, grad_weights_o, m_weights_o, v_weights_o, lr, beta1, beta2, timestep);
}

void Attention::update_adamw(float lr, float beta1, float beta2, float weight_decay) {
    timestep++;

    update_adamw_2d(weights_q, grad_weights_q, m_weights_q, v_weights_q, lr, beta1, beta2, weight_decay, timestep);
    update_adamw_2d(weights_k, grad_weights_k, m_weights_k, v_weights_k, lr, beta1, beta2, weight_decay, timestep);
    update_adamw_2d(weights_v, grad_weights_v, m_weights_v, v_weights_v, lr, beta1, beta2, weight_decay, timestep);
    update_adamw_2d(weights_o, grad_weights_o, m_weights_o, v_weights_o, lr, beta1, beta2, weight_decay, timestep);
}