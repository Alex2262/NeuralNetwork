

#ifndef NEURALNETWORK_LAYERS_H
#define NEURALNETWORK_LAYERS_H


#include <xtensor/xarray.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>
#include "../types.h"


class Layer {
public:
    virtual ~Layer() = default;

    virtual std::vector<size_t> get_input_size() const = 0;
    virtual std::vector<size_t> get_output_size() const = 0;

    virtual ActivationID get_activation_id() = 0;
    virtual xt::xarray<float> get_outputs() = 0;
    virtual xt::xarray<float> feedforward(const xt::xarray<float>& inputs, bool evaluation_mode) = 0;
    virtual xt::xarray<float> backprop(const xt::xarray<float>& delta, bool calc_delta_activation) = 0;
    virtual void update(float lr) = 0;
};


#endif //NEURALNETWORK_LAYERS_H
