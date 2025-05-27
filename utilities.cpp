
#include <string>
#include <xtensor/core/xmath.hpp>
#include <xtensor/views/xview.hpp>
#include "utilities.h"

xt::xarray<float> no_activation(const xt::xarray<float>& x) {
    return x;
}

xt::xarray<float> no_activation_derivative(const xt::xarray<float>& x) {
    return xt::ones_like(x);
}

xt::xarray<float> ReLU(const xt::xarray<float>& x) {
    return xt::maximum(x, 0.0f);
}

xt::xarray<float> ReLU_derivative(const xt::xarray<float>& x) {
    return xt::where(x > 0.0f, 1.0f, 0.0f);
}

xt::xarray<float> GELU(const xt::xarray<float>& x) {
    const float sqrt2 = std::sqrt(2.0f);
    return 0.5f * x * (1.0f + xt::erf(x / sqrt2));
}

xt::xarray<float> GELU_derivative(const xt::xarray<float>& x) {
    const float sqrt2 = std::sqrt(2.0f);
    const auto inv_sqrt2pi = 1.0f / static_cast<float>(std::sqrt(2.0f * M_PI));
    auto erf_term = xt::erf(x / sqrt2);
    auto pdf = xt::exp(-0.5f * x * x) * inv_sqrt2pi;
    return 0.5f * (1.0f + erf_term) + 0.5f * x * pdf;
}

// GELU APPROX
xt::xarray<float> GELU_approx(const xt::xarray<float>& x) {
    const auto sqrt_2_over_pi = static_cast<float>(std::sqrt(2.0f / M_PI));
    const float c = 0.044715f;
    auto x3 = x * x * x;
    auto arg = sqrt_2_over_pi * (x + c * x3);
    return 0.5f * x * (1.0f + xt::tanh(arg));
}

xt::xarray<float> GELU_approx_derivative(const xt::xarray<float>& x) {
    const auto sqrt_2_over_pi = static_cast<float>(std::sqrt(2.0f / M_PI));
    const float c = 0.044715f;

    auto x2       = x * x;
    auto x3       = x * x2;
    auto arg      = sqrt_2_over_pi * (x + c * x3);
    auto tanh_arg = xt::tanh(arg);
    auto sech2    = 1.0f - tanh_arg * tanh_arg;
    auto inner    = sqrt_2_over_pi * (1.0f + 3.0f * c * x2);

    return 0.5f * (1.0f + tanh_arg) + 0.5f * x * inner * sech2;
}

xt::xarray<float> GELU_fast(const xt::xarray<float>& x) {
    return x * sigmoid(1.702f * x);
}

xt::xarray<float> GELU_fast_derivative(const xt::xarray<float>& x) {
    auto s = sigmoid(1.702f * x);
    return s + 1.702f * x * s * (1.0f - s);
}

xt::xarray<float> sigmoid(const xt::xarray<float>& x) {
    return 1.0f / (1.0f + xt::exp(-x));
}

xt::xarray<float> sigmoid_derivative(const xt::xarray<float>& x) {
    auto s = sigmoid(x);
    return s * (1.0f - s);
}

// Assume a 2d tensor for batching
xt::xarray<float> softmax(const xt::xarray<float>& x) {
    // assume input shape: (batch_size, num_classes)

    // For numerical stability
    auto mx = xt::eval(xt::amax(x, {1}, xt::keep_dims));
    auto shifted = x - mx;

    // softmax
    auto exps = xt::exp(shifted);
    auto sum = xt::eval(xt::sum(exps, {1}, xt::keep_dims));

    return exps / sum;
}


xt::xarray<float> cross_entropy_loss(const xt::xarray<float>& probs, const xt::xarray<float>& labels) {
    auto sample_loss = -xt::sum(labels * xt::log(probs + EPSILON), {1});  // sum over classes, 1e-9 for numerical safety
    return sample_loss;
}

xt::xarray<float> MSE(const xt::xarray<float>& activation, const xt::xarray<float>& labels) {
    auto squared_error = 0.5f * xt::square(activation - labels);
    auto sample_loss = xt::sum(squared_error, {1});

    return sample_loss;
}

xt::xarray<float> MSE_derivative(const xt::xarray<float>& activation, const xt::xarray<float>& labels) {
    return activation - labels;
}

ActivationFunction get_activation_function(ActivationID activation_id) {
    switch (activation_id) {
        case ActivationID::RELU: return ReLU;
        case ActivationID::SIGMOID: return sigmoid;
        case ActivationID::SOFTMAX: return softmax;
        case ActivationID::GELU: return GELU_fast;
        default: return no_activation;
    }
}

ActivationDerivative get_activation_derivative(ActivationID activation_id) {
    switch (activation_id) {
        case ActivationID::RELU: return ReLU_derivative;
        case ActivationID::SIGMOID: return sigmoid_derivative;
        case ActivationID::SOFTMAX: return nullptr; // softmax derivative handled differently
        case ActivationID::GELU: return GELU_fast_derivative;
        default: return no_activation_derivative;
    }
}

CostFunction get_cost_function(CostID cost_id) {
    switch (cost_id) {
        case CostID::MSE:
            return [](const xt::xarray<float>& activation, const xt::xarray<float>& labels) {
                return xt::mean(MSE(activation, labels))();
            };
        case CostID::CEL:
            return [](const xt::xarray<float>& activation, const xt::xarray<float>& labels) {
                return xt::mean(cross_entropy_loss(activation, labels))();
            };
        default:
            return nullptr;
    }
}

CostDerivative get_cost_derivative(CostID cost_id) {
    switch (cost_id) {
        case CostID::MSE: return MSE_derivative;
        default: return nullptr;
    }
}

xt::xarray<float> get_output_error(const xt::xarray<float>& output, const xt::xarray<float>& activations,
                                    const xt::xarray<float>& labels, ActivationID activation_id, CostID cost_id) {
    if (activation_id == ActivationID::SOFTMAX && cost_id == CostID::CEL) {
        return activations - labels;
    }
    return get_cost_derivative(cost_id)(activations, labels) * get_activation_derivative(activation_id)(output);
}


xt::xarray<float> convert_vec_inputs(const std::vector<xt::xarray<float>>& inputs) {
    auto input_shape = inputs[0].shape();
    std::vector<size_t> new_shape = {inputs.size()};
    new_shape.insert(new_shape.end(), input_shape.begin(), input_shape.end());

    xt::xarray<float> new_inputs = xt::zeros<float>(new_shape);

    for (size_t i = 0; i < inputs.size(); i++) {
        xt::view(new_inputs, i) = inputs[i];
    }

    return new_inputs;
}



xt::xtensor<float, 2> index_3d(xt::xtensor<float, 3>& inputs, size_t index) {
    const size_t rows = inputs.shape()[1];
    const size_t cols = inputs.shape()[2];

    xt::xtensor<float, 2> out({rows, cols});
    auto* in_ptr = inputs.data() + index * rows * cols;
    auto* out_ptr = out.data();

    std::copy(in_ptr, in_ptr + rows * cols, out_ptr);

    return out;
}


void set_3d(xt::xtensor<float, 3>& inputs, xt::xtensor<float, 2>& value, size_t index) {
    const size_t rows = inputs.shape()[1];
    const size_t cols = inputs.shape()[2];

    auto* in_ptr = value.data();
    auto* out_ptr = inputs.data() + index * rows * cols;

    std::copy(in_ptr, in_ptr + rows * cols, out_ptr);
}


void update_adam_1d(xt::xtensor<float, 1>& weights, xt::xtensor<float, 1>& grad_weights,
                    xt::xtensor<float, 1>& m_weights, xt::xtensor<float, 1>& v_weights,
                    float lr, float beta1, float beta2, size_t timestep) {
    assert(weights.shape() == grad_weights.shape() && grad_weights.shape() == m_weights.shape() && m_weights.shape() == v_weights.shape());

    m_weights = beta1 * m_weights + (1.0f - beta1) * grad_weights;
    v_weights = beta2 * v_weights + (1.0f - beta2) * xt::square(grad_weights);

    xt::xtensor<float, 1> m_hat_w = m_weights / (1.0f - std::pow(beta1, timestep));
    xt::xtensor<float, 1> v_hat_w = v_weights / (1.0f - std::pow(beta2, timestep));

    weights -= lr * m_hat_w / (xt::sqrt(v_hat_w) + EPSILON);

    grad_weights.fill(0);
}

void update_adam_2d(xt::xtensor<float, 2>& weights, xt::xtensor<float, 2>& grad_weights,
                    xt::xtensor<float, 2>& m_weights, xt::xtensor<float, 2>& v_weights,
                    float lr, float beta1, float beta2, size_t timestep) {
    assert(weights.shape() == grad_weights.shape() && grad_weights.shape() == m_weights.shape() && m_weights.shape() == v_weights.shape());

    m_weights = beta1 * m_weights + (1.0f - beta1) * grad_weights;
    v_weights = beta2 * v_weights + (1.0f - beta2) * xt::square(grad_weights);

    xt::xtensor<float, 2> m_hat_w = m_weights / (1.0f - std::pow(beta1, timestep));
    xt::xtensor<float, 2> v_hat_w = v_weights / (1.0f - std::pow(beta2, timestep));

    weights -= lr * m_hat_w / (xt::sqrt(v_hat_w) + EPSILON);

    grad_weights.fill(0);
}

void update_adam_4d(xt::xtensor<float, 4>& weights, xt::xtensor<float, 4>& grad_weights,
                    xt::xtensor<float, 4>& m_weights, xt::xtensor<float, 4>& v_weights,
                    float lr, float beta1, float beta2, size_t timestep) {
    assert(weights.shape() == grad_weights.shape() && grad_weights.shape() == m_weights.shape() && m_weights.shape() == v_weights.shape());

    m_weights = beta1 * m_weights + (1.0f - beta1) * grad_weights;
    v_weights = beta2 * v_weights + (1.0f - beta2) * xt::square(grad_weights);

    xt::xtensor<float, 4> m_hat_w = m_weights / (1.0f - std::pow(beta1, timestep));
    xt::xtensor<float, 4> v_hat_w = v_weights / (1.0f - std::pow(beta2, timestep));

    weights -= lr * m_hat_w / (xt::sqrt(v_hat_w) + EPSILON);

    grad_weights.fill(0);
}


void update_adamw_1d(xt::xtensor<float, 1>& weights, xt::xtensor<float, 1>& grad_weights,
                     xt::xtensor<float, 1>& m_weights, xt::xtensor<float, 1>& v_weights,
                     float lr, float beta1, float beta2, float weight_decay, size_t timestep) {

    assert(weights.shape() == grad_weights.shape() && grad_weights.shape() == m_weights.shape() && m_weights.shape() == v_weights.shape());

    m_weights = beta1 * m_weights + (1.0f - beta1) * grad_weights;
    v_weights = beta2 * v_weights + (1.0f - beta2) * xt::square(grad_weights);

    xt::xtensor<float, 1> m_hat_w = m_weights / (1.0f - std::pow(beta1, timestep));
    xt::xtensor<float, 1> v_hat_w = v_weights / (1.0f - std::pow(beta2, timestep));

    weights *= (1.0f - lr * weight_decay);
    weights -= lr * m_hat_w / (xt::sqrt(v_hat_w) + EPSILON);

    grad_weights.fill(0);
}

void update_adamw_2d(xt::xtensor<float, 2>& weights, xt::xtensor<float, 2>& grad_weights,
                    xt::xtensor<float, 2>& m_weights, xt::xtensor<float, 2>& v_weights,
                    float lr, float beta1, float beta2, float weight_decay, size_t timestep) {
    assert(weights.shape() == grad_weights.shape() && grad_weights.shape() == m_weights.shape() && m_weights.shape() == v_weights.shape());

    m_weights = beta1 * m_weights + (1.0f - beta1) * grad_weights;
    v_weights = beta2 * v_weights + (1.0f - beta2) * xt::square(grad_weights);

    xt::xtensor<float, 2> m_hat_w = m_weights / (1.0f - std::pow(beta1, timestep));
    xt::xtensor<float, 2> v_hat_w = v_weights / (1.0f - std::pow(beta2, timestep));

    weights *= (1.0f - lr * weight_decay);
    weights -= lr * m_hat_w / (xt::sqrt(v_hat_w) + EPSILON);

    grad_weights.fill(0);
}

void update_adamw_4d(xt::xtensor<float, 4>& weights, xt::xtensor<float, 4>& grad_weights,
                    xt::xtensor<float, 4>& m_weights, xt::xtensor<float, 4>& v_weights,
                    float lr, float beta1, float beta2, float weight_decay, size_t timestep) {
    assert(weights.shape() == grad_weights.shape() && grad_weights.shape() == m_weights.shape() && m_weights.shape() == v_weights.shape());

    m_weights = beta1 * m_weights + (1.0f - beta1) * grad_weights;
    v_weights = beta2 * v_weights + (1.0f - beta2) * xt::square(grad_weights);

    xt::xtensor<float, 4> m_hat_w = m_weights / (1.0f - std::pow(beta1, timestep));
    xt::xtensor<float, 4> v_hat_w = v_weights / (1.0f - std::pow(beta2, timestep));

    weights *= (1.0f - lr * weight_decay);
    weights -= lr * m_hat_w / (xt::sqrt(v_hat_w) + EPSILON);

    grad_weights.fill(0);
}


void print_2d(xt::xtensor<float, 2>& inp) {
    for (size_t r = 0; r < inp.shape()[0]; r++) {
        std::string s;
        for (size_t c = 0; c < inp.shape()[1]; c++) {
            s += std::to_string(inp(r, c)) + " ";
        }

        std::cout << s << std::endl;
    }
}

std::string format_time(size_t secs) {
    size_t hours   = secs / 3600;
    size_t minutes = secs / 60 % 60;
    size_t seconds = secs % 60;

    std::string s = (hours   < 10 ? "0" : "") + std::to_string(hours  ) + ":"
                  + (minutes < 10 ? "0" : "") + std::to_string(minutes) + ":"
                  + (seconds < 10 ? "0" : "") + std::to_string(seconds);

    return s;
}

void save_1d(std::vector<float>& all, xt::xtensor<float, 1>& x) {
    for (size_t i = 0; i < x.shape()[0]; i++) {
        all.push_back(x(i));
    }
}

void save_2d(std::vector<float>& all, xt::xtensor<float, 2>& x) {
    for (size_t i = 0; i < x.shape()[0]; i++) {
        for (size_t j = 0; j < x.shape()[1]; j++) {
            all.push_back(x(i, j));
        }
    }
}

void save_4d(std::vector<float>& all, xt::xtensor<float, 4>& x) {
    for (size_t i = 0; i < x.shape()[0]; i++) {
        for (size_t j = 0; j < x.shape()[1]; j++) {
            for (size_t k = 0; k < x.shape()[2]; k++) {
                for (size_t c = 0; c < x.shape()[3]; c++) {
                    all.push_back(x(i, j, k, c));
                }
            }
        }
    }
}


void load_1d(xt::xtensor<float, 1>& all, xt::xtensor<float, 1>& x, size_t& index) {
    for (size_t i = 0; i < x.shape()[0]; i++) {
        x(i) = all(index);
        index++;
    }
}

void load_2d(xt::xtensor<float, 1>& all, xt::xtensor<float, 2>& x, size_t& index) {
    for (size_t i = 0; i < x.shape()[0]; i++) {
        for (size_t j = 0; j < x.shape()[1]; j++) {
            x(i, j) = all(index);
            index++;
        }
    }
}

void load_4d(xt::xtensor<float, 1>& all, xt::xtensor<float, 4>& x, size_t& index) {
    for (size_t i = 0; i < x.shape()[0]; i++) {
        for (size_t j = 0; j < x.shape()[1]; j++) {
            for (size_t k = 0; k < x.shape()[2]; k++) {
                for (size_t c = 0; c < x.shape()[3]; c++) {
                    x(i, j, k, c) = all(index);
                    index++;
                }
            }
        }
    }
}

template <typename Out>
void split(const std::string &s, char delim, Out result) {
    std::istringstream iss(s);
    std::string item;
    while (std::getline(iss, item, delim)) {
        *result++ = item;
    }
}

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, std::back_inserter(elems));
    return elems;
}