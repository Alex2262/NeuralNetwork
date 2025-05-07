
#ifndef NEURALNETWORK_TYPES_H
#define NEURALNETWORK_TYPES_H

#include <xtensor/xarray.hpp>
#include <Eigen/Dense>
#include <functional>

using ActivationFunction = std::function<xt::xarray<float>(const xt::xarray<float>&)>;
using ActivationDerivative = std::function<xt::xarray<float>(const xt::xarray<float>&)>;

using CostFunction = std::function<float(const xt::xarray<float>&, const xt::xarray<float>&)>;
using CostDerivative = std::function<xt::xarray<float>(const xt::xarray<float>&, const xt::xarray<float>&)>;

using EigenMat = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

constexpr float EPSILON = 1e-9f;

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

enum class LLM_Mode {
    CHARS,
    LOWER_CHARS,
    WORDS
};

enum class Mode {
    TRAINING,
    EVALUATION,
    INFERENCE
};

#endif //NEURALNETWORK_TYPES_H
