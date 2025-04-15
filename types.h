
#ifndef NEURALNETWORK_TYPES_H
#define NEURALNETWORK_TYPES_H

#include <xtensor/xarray.hpp>
#include <functional>

using ActivationFunction = std::function<xt::xarray<float>(const xt::xarray<float>&)>;
using ActivationDerivative = std::function<xt::xarray<float>(const xt::xarray<float>&)>;

using CostFunction = std::function<float(const xt::xarray<float>&, const xt::xarray<float>&)>;
using CostDerivative = std::function<xt::xarray<float>(const xt::xarray<float>&, const xt::xarray<float>&)>;

enum class ActivationID {
    RELU,
    SIGMOID,
    SOFTMAX,
    NONE
};

enum class CostID {
    MSE,
    CEL,
    NONE
};

#endif //NEURALNETWORK_TYPES_H
