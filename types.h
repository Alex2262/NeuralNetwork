
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
    GELU,
    NONE
};

enum class CostID {
    MSE,
    CEL,
    NONE
};

enum class LLM_MODE {
    CHARS,
    LOWER_CHARS,
    WORDS
};

#endif //NEURALNETWORK_TYPES_H
