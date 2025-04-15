
#include <string>
#include <xtensor/xmath.hpp>
#include <xtensor/xview.hpp>
#include "utilities.h"

xt::xarray<float> ReLU(const xt::xarray<float>& x) {
    return xt::maximum(x, 0.0);
}

xt::xarray<float> ReLU_derivative(const xt::xarray<float>& x) {
    return xt::where(x > 0.0, 1.0, 0.0);
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
        default: return nullptr;
    }
}

ActivationDerivative get_activation_derivative(ActivationID activation_id) {
    switch (activation_id) {
        case ActivationID::RELU: return ReLU_derivative;
        case ActivationID::SIGMOID: return sigmoid_derivative;
        default: return nullptr;  // softmax derivative handled differently
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
    xt::xtensor<float, 2> out = xt::zeros<float>({inputs.shape()[1], inputs.shape()[2]});

    for (size_t i = 0; i < inputs.shape()[1]; i++) {
        for (size_t j = 0; j < inputs.shape()[2]; j++) {
            out(i, j) = inputs(index, i, j);
        }
    }

    return out;
}


void set_3d(xt::xtensor<float, 3>& inputs, xt::xtensor<float, 2>& value, size_t index) {
    for (size_t i = 0; i < inputs.shape()[1]; i++) {
        for (size_t j = 0; j < inputs.shape()[2]; j++) {
            inputs(index, i, j) = value(i, j);
        }
    }
}


void update_adam_1d(xt::xtensor<float, 1>& weights, xt::xtensor<float, 1>& grad_weights,
                    xt::xtensor<float, 1>& m_weights, xt::xtensor<float, 1>& v_weights,
                    float lr, float beta1, float beta2, float epsilon, size_t timestep) {

    m_weights = beta1 * m_weights + (1.0f - beta1) * grad_weights;
    v_weights = beta2 * v_weights + (1.0f - beta2) * xt::square(grad_weights);

    xt::xtensor<float, 1> m_hat_w = m_weights / (1.0f - std::pow(beta1, timestep));
    xt::xtensor<float, 1> v_hat_w = v_weights / (1.0f - std::pow(beta2, timestep));

    weights -= lr * m_hat_w / (xt::sqrt(v_hat_w) + epsilon);

    grad_weights.fill(0);
}

void update_adam_2d(xt::xtensor<float, 2>& weights, xt::xtensor<float, 2>& grad_weights,
                    xt::xtensor<float, 2>& m_weights, xt::xtensor<float, 2>& v_weights,
                    float lr, float beta1, float beta2, float epsilon, size_t timestep) {

    m_weights = beta1 * m_weights + (1.0f - beta1) * grad_weights;
    v_weights = beta2 * v_weights + (1.0f - beta2) * xt::square(grad_weights);

    xt::xtensor<float, 2> m_hat_w = m_weights / (1.0f - std::pow(beta1, timestep));
    xt::xtensor<float, 2> v_hat_w = v_weights / (1.0f - std::pow(beta2, timestep));

    weights -= lr * m_hat_w / (xt::sqrt(v_hat_w) + epsilon);

    grad_weights.fill(0);
}

void update_adam_4d(xt::xtensor<float, 4>& weights, xt::xtensor<float, 4>& grad_weights,
                    xt::xtensor<float, 4>& m_weights, xt::xtensor<float, 4>& v_weights,
                    float lr, float beta1, float beta2, float epsilon, size_t timestep) {

    m_weights = beta1 * m_weights + (1.0f - beta1) * grad_weights;
    v_weights = beta2 * v_weights + (1.0f - beta2) * xt::square(grad_weights);

    xt::xtensor<float, 4> m_hat_w = m_weights / (1.0f - std::pow(beta1, timestep));
    xt::xtensor<float, 4> v_hat_w = v_weights / (1.0f - std::pow(beta2, timestep));

    weights -= lr * m_hat_w / (xt::sqrt(v_hat_w) + epsilon);

    grad_weights.fill(0);
}