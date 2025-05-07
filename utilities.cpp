
#include <string>
#include <xtensor/xmath.hpp>
#include <xtensor/xview.hpp>
#include <Eigen/Core>
#include "utilities.h"

xt::xarray<float> no_activation(const xt::xarray<float>& x) {
    return x;
}

xt::xarray<float> no_activation_derivative(const xt::xarray<float>& x) {
    return xt::ones_like(x);
}

xt::xarray<float> ReLU(const xt::xarray<float>& x) {
    return xt::maximum(x, 0.0);
}

xt::xarray<float> ReLU_derivative(const xt::xarray<float>& x) {
    return xt::where(x > 0.0, 1.0, 0.0);
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
    auto x3 = xt::pow(x, 3);
    auto arg = sqrt_2_over_pi * (x + c * x3);
    return 0.5f * x * (1.0f + xt::tanh(arg));
}

xt::xarray<float> GELU_approx_derivative(const xt::xarray<float>& x) {
    const auto sqrt_2_over_pi = static_cast<float>(std::sqrt(2.0f / M_PI));
    const float c = 0.044715f;

    auto x2       = xt::pow(x, 2);
    auto x3       = x * x2;
    auto arg      = sqrt_2_over_pi * (x + c * x3);
    auto tanh_arg = xt::tanh(arg);
    auto sech2    = 1.0f - tanh_arg * tanh_arg;
    auto inner    = sqrt_2_over_pi * (1.0f + 3.0f * c * x2);

    return 0.5f * (1.0f + tanh_arg) + 0.5f * x * inner * sech2;
}

xt::xarray<float> sigmoid(const xt::xarray<float>& x) {
    return 1.0 / (1.0 + xt::exp(-x));
}

xt::xarray<float> sigmoid_derivative(const xt::xarray<float>& x) {
    auto s = sigmoid(x);
    return s * (1.0 - s);
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
    auto sample_loss = -xt::sum(labels * xt::log(probs + 1e-9), {1});  // sum over classes, 1e-9 for numerical safety
    return sample_loss;
}

xt::xarray<float> MSE(const xt::xarray<float>& activation, const xt::xarray<float>& labels) {
    auto squared_error = 0.5 * xt::square(activation - labels);
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
        case ActivationID::GELU: return GELU_approx;
        default: return no_activation;
    }
}

ActivationDerivative get_activation_derivative(ActivationID activation_id) {
    switch (activation_id) {
        case ActivationID::RELU: return ReLU_derivative;
        case ActivationID::SIGMOID: return sigmoid_derivative;
        case ActivationID::SOFTMAX: return nullptr; // softmax derivative handled differently
        case ActivationID::GELU: return GELU_approx_derivative;
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

xt::xtensor<float, 2> eigen_dot(const xt::xtensor<float, 2>& p_A, const xt::xtensor<float, 2>& p_B) {
    assert(p_A.shape()[1] == p_B.shape()[0]);

    const xt::xtensor<float, 2> A = xt::eval(p_A);
    const xt::xtensor<float, 2> B = xt::eval(p_B);

    EigenMat A_eigen = Eigen::Map<const EigenMat>(A.data(), A.shape()[0], A.shape()[1]);
    EigenMat B_eigen = Eigen::Map<const EigenMat>(B.data(), B.shape()[0], B.shape()[1]);

    EigenMat C_eigen = A_eigen * B_eigen;

    xt::xtensor<float, 2> C = xt::empty<float>({A.shape()[0], B.shape()[1]});
    std::copy(C_eigen.data(), C_eigen.data() + C.size(), C.data());

    return C;
}

