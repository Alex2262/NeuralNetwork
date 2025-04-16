

#ifndef NEURALNETWORK_NEURAL_NETWORK_H
#define NEURALNETWORK_NEURAL_NETWORK_H

#include <string>
#include "utilities.h"
#include "layers/layers.h"

class NeuralNetwork {

private:
    std::vector<size_t> input_size;
    CostID cost_id;
    std::vector<std::unique_ptr<Layer>> layers;

public:
    NeuralNetwork(std::vector<size_t> p_input_size, CostID p_cost_id);

    size_t get_num_layers() { return layers.size(); }
    Layer* get_layer(size_t index);

    xt::xarray<float> feedforward(const xt::xarray<float>& inputs, bool evaluation_mode);

    void backprop(const xt::xarray<float>& inputs, const xt::xarray<float>& labels);

    void update(const xt::xarray<float>& inputs,
                const xt::xarray<float>& labels,
                float lr);

    void update_adam(const xt::xarray<float>& inputs,
                     const xt::xarray<float>& labels,
                     float lr, float beta1, float beta2, float epsilon);

    float evaluate(const xt::xarray<float>& inputs, const xt::xarray<float>& labels);

    void SGD(const std::vector<xt::xarray<float>>& training_inputs,
             const std::vector<xt::xarray<float>>& training_labels,
             const std::vector<xt::xarray<float>>& test_inputs,
             const std::vector<xt::xarray<float>>& test_labels,
             size_t epochs, size_t mini_batch_size, float lr);

    void Adam(const std::vector<xt::xarray<float>>& training_inputs,
              const std::vector<xt::xarray<float>>& training_labels,
              const std::vector<xt::xarray<float>>& test_inputs,
              const std::vector<xt::xarray<float>>& test_labels,
              size_t epochs, size_t mini_batch_size, float lr, float beta1, float beta2, float epsilon);

    template <typename LayerType, typename... Args>
    void add_layer(Args&&... args) {
        std::vector<size_t> prev_output_size = layers.empty() ? input_size : layers.back()->get_output_size();

        layers.push_back(std::make_unique<LayerType>(prev_output_size, std::forward<Args>(args)...));

        auto output_size = layers.back()->get_output_size();

        std::string f_input_size = std::to_string(prev_output_size[0]);
        std::string f_output_size = std::to_string(output_size[0]);

        for (int i = 1; i < prev_output_size.size(); i++) f_input_size += ", " + std::to_string(prev_output_size[i]);
        for (int i = 1; i < output_size.size(); i++) f_output_size += ", " + std::to_string(output_size[i]);

        std::cout << "Added Layer with input size (" << f_input_size
                  << ") and output size (" << f_output_size << ")" << std::endl;
    }
};

#endif //NEURALNETWORK_NEURAL_NETWORK_H
