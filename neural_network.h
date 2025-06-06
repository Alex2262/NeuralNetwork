

#ifndef NEURALNETWORK_NEURAL_NETWORK_H
#define NEURALNETWORK_NEURAL_NETWORK_H

#include <string>
#include "utilities.h"
#include "layers/layers.h"

struct TrainInfo {
    size_t timestep = 0;
    size_t mini_batch_size = 0;
    size_t num_epochs = 0;
    size_t super_batch_size = 0;
    size_t num_super_batch = 0;
    float lr = 0;
    float beta1 = 0;
    float beta2 = 0;
    float weight_decay = 0;
    std::string save_prefix;

    size_t current_epoch = 0;
    size_t current_super_batch = 0;
};

float get_lr_linear_warmup_cosine_decay(size_t current_step, size_t end_step, float base_lr, float warmup_ratio = 0.05f);

class NeuralNetwork {

private:
    std::vector<size_t> input_size;
    CostID cost_id;
    std::vector<std::unique_ptr<Layer>> layers;

    size_t num_params;
    TrainInfo train_info;

public:
    NeuralNetwork(std::vector<size_t> p_input_size, CostID p_cost_id);

    size_t get_num_layers() { return layers.size(); }
    size_t get_num_params() const { return num_params; }

    Layer* get_layer(size_t index);

    xt::xarray<float> feedforward(const xt::xarray<float>& inputs, Mode mode);

    void backprop(const xt::xarray<float>& inputs, const xt::xarray<float>& labels);

    void update(const xt::xarray<float>& inputs,
                const xt::xarray<float>& labels,
                float lr);

    void update_adam(const xt::xarray<float>& inputs,
                     const xt::xarray<float>& labels,
                     float lr, float beta1, float beta2);

    void update_adamw(const xt::xarray<float>& inputs,
                      const xt::xarray<float>& labels,
                      float lr, float beta1, float beta2, float weight_decay);

    float evaluate(const xt::xarray<float>& inputs, const xt::xarray<float>& labels);
    float loss(const xt::xarray<float>& inputs, const xt::xarray<float>& labels);

    void SGD(const std::vector<xt::xarray<float>>& training_inputs,
             const std::vector<xt::xarray<float>>& training_labels,
             const std::vector<xt::xarray<float>>& test_inputs,
             const std::vector<xt::xarray<float>>& test_labels);

    void Adam(const std::vector<xt::xarray<float>>& training_inputs,
              const std::vector<xt::xarray<float>>& training_labels,
              const std::vector<xt::xarray<float>>& test_inputs,
              const std::vector<xt::xarray<float>>& test_labels);

    void AdamW(const std::vector<xt::xarray<float>>& training_inputs,
               const std::vector<xt::xarray<float>>& training_labels,
               const std::vector<xt::xarray<float>>& test_inputs,
               const std::vector<xt::xarray<float>>& test_labels);

    void save(std::string& file_prefix);
    void load(std::string& file_prefix);

    TrainInfo* get_train_info() { return &train_info; }
    void set_train_info(TrainInfo& p_train_info);

    template <typename LayerType, typename... Args>
    void add_layer(Args&&... args) {
        std::vector<size_t> prev_output_size = layers.empty() ? input_size : layers.back()->get_output_size();

        layers.push_back(std::make_unique<LayerType>(prev_output_size, std::forward<Args>(args)...));

        auto output_size = layers.back()->get_output_size();

        std::string f_input_size = std::to_string(prev_output_size[0]);
        std::string f_output_size = std::to_string(output_size[0]);

        for (int i = 1; i < prev_output_size.size(); i++) f_input_size += ", " + std::to_string(prev_output_size[i]);
        for (int i = 1; i < output_size.size(); i++) f_output_size += ", " + std::to_string(output_size[i]);

        std::cout << "Added " << layers.back()->get_name() << " Layer with input size (" << f_input_size
                  << ") and output size (" << f_output_size << ")" << std::endl;

        num_params += layers.back()->get_num_params();
    }
};

#endif //NEURALNETWORK_NEURAL_NETWORK_H
