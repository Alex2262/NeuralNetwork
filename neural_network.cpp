
#include <iostream>
#include <iomanip>
#include <random>
#include <filesystem>
#include <xtensor/misc/xsort.hpp>
#include <xtensor/io/xnpy.hpp>
#include "neural_network.h"


float get_lr_linear_warmup_cosine_decay(size_t current_step, size_t end_step, float base_lr, float warmup_ratio) {
    auto warmup_steps = static_cast<size_t>(warmup_ratio * static_cast<float>(end_step));

    // Linear Warmup
    if (current_step < warmup_steps) {
        return base_lr * static_cast<float>(current_step) / static_cast<float>(warmup_steps);
    }

    // Cosine Decay
    else if (current_step <= end_step) {
        float progress = static_cast<float>(current_step - warmup_steps) / static_cast<float>(end_step - warmup_steps);
        return base_lr * 0.5f * (1.0f + static_cast<float>(std::cos(M_PI * progress)));
    }

    return 0.0f;
}


NeuralNetwork::NeuralNetwork(std::vector<size_t> p_input_size, CostID p_cost_id) {
    input_size = p_input_size;
    cost_id = p_cost_id;
    num_params = 0;
}

Layer* NeuralNetwork::get_layer(size_t index) {
    if (index >= layers.size()) return nullptr;
    return layers[index].get();
}

xt::xarray<float> NeuralNetwork::feedforward(const xt::xarray<float>& inputs, Mode mode) {
    xt::xarray<float> activation = inputs;
    for (std::unique_ptr<Layer>& layer : layers) {
        activation = layer->feedforward(activation, mode);
    }

    return activation;
}

void NeuralNetwork::backprop(const xt::xarray<float>& inputs, const xt::xarray<float>& labels) {
    xt::xarray<float> activation = feedforward(inputs, Mode::TRAINING);
    xt::xarray<float> output = layers.back()->get_outputs();
    xt::xarray<float> delta = get_output_error(output, activation, labels, layers.back()->get_activation_id(), cost_id);

    for (int i = static_cast<int>(layers.size() - 1); i >= 0; i--) {
        delta = layers[i]->backprop(delta, i != layers.size() - 1);
    }
}

void NeuralNetwork::update(const xt::xarray<float>& inputs,
                           const xt::xarray<float>& labels,
                           float lr) {
    backprop(inputs, labels);

    for (std::unique_ptr<Layer>& layer : layers) {
        layer->update(lr);
    }
}

void NeuralNetwork::update_adam(const xt::xarray<float>& inputs,
                                const xt::xarray<float>& labels,
                                float lr, float beta1, float beta2) {
    train_info.timestep++;
    backprop(inputs, labels);

    for (std::unique_ptr<Layer>& layer : layers) {
        layer->update_adam(lr, beta1, beta2, train_info.timestep);
    }
}

void NeuralNetwork::update_adamw(const xt::xarray<float>& inputs,
                                 const xt::xarray<float>& labels,
                                 float lr, float beta1, float beta2, float weight_decay) {
    train_info.timestep++;
    backprop(inputs, labels);

    for (std::unique_ptr<Layer>& layer : layers) {
        layer->update_adamw(lr, beta1, beta2, weight_decay, train_info.timestep);
    }
}

float NeuralNetwork::evaluate(const xt::xarray<float>& inputs, const xt::xarray<float>& labels) {
    xt::xarray<float> test_labels = labels;
    size_t batch_size = inputs.shape()[0];

    xt::xarray<float> activations = feedforward(inputs, Mode::EVALUATION);

    if (labels.shape().size() == 3) {
        batch_size = activations.shape()[0] * activations.shape()[1];
        activations = xt::reshape_view(activations, {batch_size, activations.shape()[2]});
        test_labels = xt::reshape_view(test_labels, {batch_size, test_labels.shape()[2]});
    }

    xt::xarray<size_t> pred_indices = xt::argmax(activations, 1);
    xt::xarray<size_t> true_indices = xt::argmax(test_labels, 1);

    int correct = 0;

    for (size_t i = 0; i < batch_size; i++) {
        if (pred_indices(i) == true_indices(i)) {
            correct++;
        }
    }

    return static_cast<float>(correct) / static_cast<float>(batch_size);
}


float NeuralNetwork::loss(const xt::xarray<float>& inputs, const xt::xarray<float>& labels) {
    auto cost_function = get_cost_function(cost_id);
    xt::xarray<float> activations = feedforward(inputs, Mode::EVALUATION);
    xt::xarray<float> test_labels = labels;

    if (labels.shape().size() == 3) {
        size_t batch_size = activations.shape()[0] * activations.shape()[1];
        activations = xt::reshape_view(activations, {batch_size, activations.shape()[2]});
        test_labels = xt::reshape_view(test_labels, {batch_size, test_labels.shape()[2]});
    }

    return cost_function(activations, test_labels);
}

void NeuralNetwork::SGD(const std::vector<xt::xarray<float>>& training_inputs,
                        const std::vector<xt::xarray<float>>& training_labels,
                        const std::vector<xt::xarray<float>>& test_inputs,
                        const std::vector<xt::xarray<float>>& test_labels) {

    size_t epochs = train_info.num_epochs;
    size_t mini_batch_size = train_info.mini_batch_size;
    size_t start_epoch = train_info.current_epoch;

    float lr = train_info.lr;

    if (epochs == 0 || mini_batch_size == 0 || lr == 0) {
        throw std::runtime_error("Set SGD Train Info: num_epochs | mini_batch_size | lr");
    }

    auto conv_test_inputs = convert_vec_inputs(test_inputs);
    auto conv_test_labels = convert_vec_inputs(test_labels);

    size_t n = training_inputs.size();

    std::random_device rd;
    std::mt19937 g(rd());

    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    for (size_t epoch = start_epoch; epoch < epochs; epoch++) {
        auto start_time = std::chrono::high_resolution_clock::now();
        std::shuffle(indices.begin(), indices.end(), g);

        size_t num_batches = (n + mini_batch_size - 1) / mini_batch_size;

        for (size_t batch_index = 0; batch_index < num_batches; batch_index++) {
            std::vector<xt::xarray<float>> batch_inputs;
            std::vector<xt::xarray<float>> batch_labels;

            size_t start = batch_index * mini_batch_size;
            size_t end = std::min(start + mini_batch_size, n);

            for (size_t i = start; i < end; i++) {
                size_t idx = indices[i];
                batch_inputs.push_back(training_inputs[idx]);
                batch_labels.push_back(training_labels[idx]);
            }

            xt::xarray<float> inputs = convert_vec_inputs(batch_inputs);
            xt::xarray<float> labels = convert_vec_inputs(batch_labels);

            update(inputs, labels, lr);

            auto curr = std::chrono::high_resolution_clock::now();
            float elapsed_time = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(curr - start_time).count()) / 1000.0f;
            float avg_time_per_batch = static_cast<float>(elapsed_time) / static_cast<float>(batch_index + 1);
            size_t remaining_batches = static_cast<int>(num_batches - (batch_index + 1));
            float estimated_remaining_time = static_cast<float>(remaining_batches) * avg_time_per_batch;

            float percent_complete = (static_cast<float>(batch_index + 1) /
                                      static_cast<float>(num_batches)) * 100;

            std::cout << "\rEpoch #" << (epoch + 1)
                      << " progress: " << std::fixed << std::setprecision(4) << percent_complete << "% complete | "
                      << "Estimated time remaining: " << format_time(static_cast<size_t>(estimated_remaining_time));

            std::fflush(stdout);
        }

        float current_accuracy = evaluate(conv_test_inputs, conv_test_labels);

        train_info.current_epoch = epoch;
        std::cout << "\rEpoch #" << (epoch + 1) << " | Accuracy: " << current_accuracy << std::endl;
    }
}



void NeuralNetwork::Adam(const std::vector<xt::xarray<float>>& training_inputs,
                         const std::vector<xt::xarray<float>>& training_labels,
                         const std::vector<xt::xarray<float>>& test_inputs,
                         const std::vector<xt::xarray<float>>& test_labels) {

    size_t epochs = train_info.num_epochs;
    size_t mini_batch_size = train_info.mini_batch_size;
    size_t start_epoch = train_info.current_epoch;

    float lr = train_info.lr;
    float beta1 = train_info.beta1;
    float beta2 = train_info.beta2;

    if (epochs == 0 || mini_batch_size == 0 || lr == 0 || beta1 == 0 || beta2 == 0) {
        throw std::runtime_error("Set Adam Train Info: num_epochs | mini_batch_size | lr | beta1 | beta2");
    }

    auto conv_test_inputs = convert_vec_inputs(test_inputs);
    auto conv_test_labels = convert_vec_inputs(test_labels);

    size_t n = training_inputs.size();

    std::random_device rd;
    std::mt19937 g(rd());

    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    for (size_t epoch = start_epoch; epoch <= epochs; epoch++) {
        auto start_time = std::chrono::high_resolution_clock::now();
        std::shuffle(indices.begin(), indices.end(), g);

        size_t num_batches = (n + mini_batch_size - 1) / mini_batch_size;

        for (size_t batch_index = 0; batch_index < num_batches; batch_index++) {
            std::vector<xt::xarray<float>> batch_inputs;
            std::vector<xt::xarray<float>> batch_labels;

            size_t start = batch_index * mini_batch_size;
            size_t end = std::min(start + mini_batch_size, n);

            for (size_t i = start; i < end; i++) {
                size_t idx = indices[i];
                batch_inputs.push_back(training_inputs[idx]);
                batch_labels.push_back(training_labels[idx]);
            }

            xt::xarray<float> inputs = convert_vec_inputs(batch_inputs);
            xt::xarray<float> labels = convert_vec_inputs(batch_labels);

            update_adam(inputs, labels, lr, beta1, beta2);

            auto curr = std::chrono::high_resolution_clock::now();
            float elapsed_time = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(curr - start_time).count()) / 1000.0f;
            float avg_time_per_batch = static_cast<float>(elapsed_time) / static_cast<float>(batch_index + 1);
            size_t remaining_batches = static_cast<int>(num_batches - (batch_index + 1));
            float estimated_remaining_time = static_cast<float>(remaining_batches) * avg_time_per_batch;

            float percent_complete = (static_cast<float>(batch_index + 1) /
                                      static_cast<float>(num_batches)) * 100;

            std::cout << "\rEpoch #" << (epoch + 1)
                      << " progress: " << std::fixed << std::setprecision(4) << percent_complete << "% complete | "
                      << "Estimated time remaining: " << format_time(static_cast<size_t>(estimated_remaining_time));

            std::fflush(stdout);
        }

        float current_accuracy = evaluate(conv_test_inputs, conv_test_labels);

        train_info.current_epoch = epoch;
        std::cout << "\rEpoch #" << (epoch + 1) << " | Accuracy: " << current_accuracy << std::endl;
    }
}

void NeuralNetwork::AdamW(const std::vector<xt::xarray<float>>& training_inputs,
                          const std::vector<xt::xarray<float>>& training_labels,
                          const std::vector<xt::xarray<float>>& test_inputs,
                          const std::vector<xt::xarray<float>>& test_labels) {

    size_t epochs = train_info.num_epochs;
    size_t mini_batch_size = train_info.mini_batch_size;
    size_t start_epoch = train_info.current_epoch + 1;

    float lr = train_info.lr;
    float beta1 = train_info.beta1;
    float beta2 = train_info.beta2;
    float weight_decay = train_info.weight_decay;

    if (epochs == 0 || mini_batch_size == 0 || lr == 0 || beta1 == 0 || beta2 == 0 || weight_decay == 0) {
        throw std::runtime_error("Set AdamW Train Info: num_epochs | mini_batch_size | lr | beta1 | beta2 | weight_decay");
    }

    auto conv_test_inputs = convert_vec_inputs(test_inputs);
    auto conv_test_labels = convert_vec_inputs(test_labels);

    size_t n = training_inputs.size();

    std::random_device rd;
    std::mt19937 g(rd());

    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    for (size_t epoch = start_epoch; epoch <= epochs; epoch++) {
        auto start_time = std::chrono::high_resolution_clock::now();
        std::shuffle(indices.begin(), indices.end(), g);

        size_t num_batches = (n + mini_batch_size - 1) / mini_batch_size;

        for (size_t batch_index = 0; batch_index < num_batches; batch_index++) {
            std::vector<xt::xarray<float>> batch_inputs;
            std::vector<xt::xarray<float>> batch_labels;

            size_t start = batch_index * mini_batch_size;
            size_t end = std::min(start + mini_batch_size, n);

            for (size_t i = start; i < end; i++) {
                size_t idx = indices[i];
                batch_inputs.push_back(training_inputs[idx]);
                batch_labels.push_back(training_labels[idx]);
            }

            xt::xarray<float> inputs = convert_vec_inputs(batch_inputs);
            xt::xarray<float> labels = convert_vec_inputs(batch_labels);

            update_adam(inputs, labels, lr, beta1, beta2);

            auto curr = std::chrono::high_resolution_clock::now();
            float elapsed_time = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(curr - start_time).count()) / 1000.0f;
            float avg_time_per_batch = static_cast<float>(elapsed_time) / static_cast<float>(batch_index + 1);
            size_t remaining_batches = static_cast<int>(num_batches - (batch_index + 1));
            float estimated_remaining_time = static_cast<float>(remaining_batches) * avg_time_per_batch;

            float percent_complete = (static_cast<float>(batch_index + 1) /
                                      static_cast<float>(num_batches)) * 100;

            std::cout << "\rEpoch #" << epoch
                      << " progress: " << std::fixed << std::setprecision(4) << percent_complete << "% complete | "
                      << "Estimated time remaining: " << format_time(static_cast<size_t>(estimated_remaining_time));

            std::fflush(stdout);
        }

        float current_accuracy = evaluate(conv_test_inputs, conv_test_labels);

        std::cout << "\rEpoch #" << epoch << " | Accuracy: " << current_accuracy << std::endl;

        std::string save_prefix = train_info.save_prefix + "_epoch_" + std::to_string(epoch);

        train_info.current_epoch = epoch;
        save(save_prefix);
    }
}


void NeuralNetwork::save(std::string& file_prefix) {
    std::string npy_file_name = file_prefix + "_weights.npy";
    std::string info_file_name = file_prefix + "_info.txt";

    std::vector<float> all;
    for (std::unique_ptr<Layer>& layer : layers) {
        layer->save_weights(all);
    }

    xt::xarray<float> data = xt::zeros<float>({all.size()});
    for (size_t i = 0; i < all.size(); i++) data(i) = all[i];
    xt::dump_npy(npy_file_name, data);

    std::ofstream info_file;
    info_file.open(info_file_name);
    if (!info_file.is_open()) {
        throw std::runtime_error("Could not open " + info_file_name);
    }

    info_file << train_info.timestep << std::endl;
    info_file << train_info.mini_batch_size << std::endl;
    info_file << train_info.num_epochs << std::endl;
    info_file << train_info.super_batch_size << std::endl;
    info_file << train_info.num_super_batch << std::endl;
    info_file << train_info.current_epoch << std::endl;
    info_file << train_info.current_super_batch << std::endl;

    info_file << train_info.lr << std::endl;
    info_file << train_info.beta1 << std::endl;
    info_file << train_info.beta2 << std::endl;
    info_file << train_info.weight_decay << std::endl;

    info_file.close();
}

void NeuralNetwork::load(std::string& file_prefix) {
    std::string npy_file_name = file_prefix + "_weights.npy";
    std::string info_file_name = file_prefix + "_info.txt";

    xt::xtensor<float, 1> data = xt::load_npy<float>(npy_file_name);

    size_t index = 0;
    for (std::unique_ptr<Layer>& layer : layers) {
        layer->load_weights(data, index);
        layer->zero_grad();
    }
    if (index != data.size()) {
        throw std::runtime_error("Weight file size mismatch when loading " + npy_file_name);
    }

    std::ifstream info_file;
    info_file.open(info_file_name);
    if (!info_file.is_open()) {
        throw std::runtime_error("Could not open " + info_file_name);
    }

    info_file >> train_info.timestep;
    info_file >> train_info.mini_batch_size;
    info_file >> train_info.num_epochs;
    info_file >> train_info.super_batch_size;
    info_file >> train_info.num_super_batch;
    info_file >> train_info.current_epoch;
    info_file >> train_info.current_super_batch;

    info_file >> train_info.lr;
    info_file >> train_info.beta1;
    info_file >> train_info.beta2;
    info_file >> train_info.weight_decay;

    info_file.close();
}


void NeuralNetwork::set_train_info(TrainInfo& p_train_info) {
    if (p_train_info.timestep != 0) train_info.timestep = p_train_info.timestep;
    if (p_train_info.mini_batch_size != 0) train_info.mini_batch_size = p_train_info.mini_batch_size;
    if (p_train_info.num_epochs != 0) train_info.num_epochs = p_train_info.num_epochs;
    if (p_train_info.super_batch_size != 0) train_info.super_batch_size = p_train_info.super_batch_size;
    if (p_train_info.num_super_batch != 0) train_info.num_super_batch = p_train_info.num_super_batch;

    if (p_train_info.lr != 0) train_info.lr = p_train_info.lr;
    if (p_train_info.beta1 != 0) train_info.beta1 = p_train_info.beta1;
    if (p_train_info.beta2 != 0) train_info.beta2 = p_train_info.beta2;
    if (p_train_info.weight_decay != 0) train_info.weight_decay = p_train_info.weight_decay;
    if (!p_train_info.save_prefix.empty()) train_info.save_prefix = p_train_info.save_prefix;

    if (p_train_info.current_epoch != 0) train_info.current_epoch = p_train_info.current_epoch;
    if (p_train_info.current_super_batch != 0) train_info.current_super_batch = p_train_info.current_super_batch;
}
