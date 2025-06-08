

#ifndef NEURALNETWORK_LAYERS_H
#define NEURALNETWORK_LAYERS_H


#include <xtensor/containers/xarray.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/views/xview.hpp>
#include "../types.h"


class Layer {
public:
    xt::xarray<float> res_delta = 0;

    void set_res_delta(const xt::xarray<float>& delta) { res_delta = delta; };

    virtual ~Layer() = default;

    virtual std::string get_name() const = 0;
    virtual std::vector<size_t> get_input_size() const = 0;
    virtual std::vector<size_t> get_output_size() const = 0;
    virtual size_t get_num_params() const = 0;

    virtual ActivationID get_activation_id() = 0;
    virtual xt::xarray<float> get_outputs() = 0;
    virtual xt::xarray<float> get_activations() = 0;
    virtual xt::xarray<float> feedforward(const xt::xarray<float>& inputs, Mode mode) = 0;
    virtual xt::xarray<float> backprop(const xt::xarray<float>& p_delta, bool calc_delta_activation) = 0;

    virtual void update(float lr) = 0;
    virtual void update_adam(float lr, float beta1, float beta2, size_t timestep) = 0;
    virtual void update_adamw(float lr, float beta1, float beta2, float weight_decay, size_t timestep) = 0;

    virtual void save_weights(std::vector<float>& all) = 0;
    virtual void load_weights(xt::xtensor<float, 1>& all, size_t& index) = 0;

    // Clear gradient buffers and residual deltas. Layers that store
    // trainable parameters should override this if they keep gradient
    // tensors between updates.
    virtual void zero_grad() { res_delta = 0; }
};


#endif //NEURALNETWORK_LAYERS_H
