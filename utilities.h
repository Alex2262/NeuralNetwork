
#ifndef NEURALNETWORK_UTILITIES_H
#define NEURALNETWORK_UTILITIES_H

#include <numeric>
#include <functional>
#include <xtensor/xarray.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xtensor.hpp>
#include "types.h"

xt::xarray<float> no_activation(const xt::xarray<float>& x);
xt::xarray<float> no_activation_derivative(const xt::xarray<float>& x);

xt::xarray<float> ReLU(const xt::xarray<float>& x);
xt::xarray<float> ReLU_derivative(const xt::xarray<float>& x);
xt::xarray<float> sigmoid(const xt::xarray<float>& x);
xt::xarray<float> sigmoid_derivative(const xt::xarray<float>& x);
xt::xarray<float> softmax(const xt::xarray<float>& x);

xt::xarray<float> cross_entropy_loss(const xt::xarray<float>& probs, const xt::xarray<float>& label);
xt::xarray<float> MSE(const xt::xarray<float>& activation, const xt::xarray<float>& labels);
xt::xarray<float> MSE_derivative(const xt::xarray<float>& activation, const xt::xarray<float>& labels);

ActivationFunction get_activation_function(ActivationID activation_id);
ActivationDerivative get_activation_derivative(ActivationID activation_id);
CostFunction get_cost_function(CostID cost_id);
CostDerivative get_cost_derivative(CostID cost_id);

xt::xarray<float> get_output_error(const xt::xarray<float>& output, const xt::xarray<float>& activations,
                                    const xt::xarray<float>& labels, ActivationID activation_id, CostID cost_id);

xt::xarray<float> convert_vec_inputs(const std::vector<xt::xarray<float>>& inputs);

xt::xtensor<float, 2> index_3d(xt::xtensor<float, 3>& inputs, size_t index);
void set_3d(xt::xtensor<float, 3>& inputs, xt::xtensor<float, 2>& value, size_t index);

void update_adam_1d(xt::xtensor<float, 1>& weights, xt::xtensor<float, 1>& grad_weights,
                    xt::xtensor<float, 1>& m_weights, xt::xtensor<float, 1>& v_weights,
                    float lr, float beta1, float beta2, float epsilon, size_t timestep);

void update_adam_2d(xt::xtensor<float, 2>& weights, xt::xtensor<float, 2>& grad_weights,
                    xt::xtensor<float, 2>& m_weights, xt::xtensor<float, 2>& v_weights,
                    float lr, float beta1, float beta2, float epsilon, size_t timestep);

void update_adam_4d(xt::xtensor<float, 4>& weights, xt::xtensor<float, 4>& grad_weights,
                    xt::xtensor<float, 4>& m_weights, xt::xtensor<float, 4>& v_weights,
                    float lr, float beta1, float beta2, float epsilon, size_t timestep);

template <typename Shape>
void print_shape(const Shape& shape) {
    std::string s;
    s += "(";
    s += std::to_string(shape[0]);

    for (int i = 1; i < shape.size(); i++) s += ", " + std::to_string(shape[i]);

    s += ")";

    std::cout << s << std::endl;
}


template <typename Shape>
std::vector<size_t> unravel_index(size_t flat_index, const Shape& shape) {
    size_t ndim = shape.size();
    std::vector<size_t> suffix_products(ndim, 1);
    std::vector<size_t> indices(ndim);

    for (int i = static_cast<int>(ndim) - 2; i >= 0; i--) {
        suffix_products[i] = suffix_products[i + 1] * shape[i + 1];
    }

    for (size_t i = 0; i < ndim; i++) {
        indices[i] = flat_index / suffix_products[i];
        flat_index %= suffix_products[i];
    }

    return indices;
}

template <typename Iterable>
auto prod(const Iterable& container, size_t start, size_t end) {
    using ValueType = std::decay_t<decltype(*std::begin(container))>;

    auto it_start = std::begin(container);
    std::advance(it_start, start);
    auto it_end = std::begin(container);
    std::advance(it_end, end + 1);

    return std::accumulate(it_start, it_end, static_cast<ValueType>(1), std::multiplies<ValueType>());
}

#endif //NEURALNETWORK_UTILITIES_H
