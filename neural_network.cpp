
#include <iostream>
#include <iomanip>
#include <random>
#include <xtensor/xsort.hpp>
#include "neural_network.h"

NeuralNetwork::NeuralNetwork(std::vector<size_t> p_input_size, CostID p_cost_id) {
    input_size = p_input_size;
    cost_id = p_cost_id;
}

Layer* NeuralNetwork::get_layer(size_t index) {
    if (index >= layers.size()) return nullptr;
    return layers[index].get();
}

xt::xarray<float> NeuralNetwork::feedforward(const xt::xarray<float>& inputs, bool evaluation_mode) {
    xt::xarray<float> activation = inputs;
    for (std::unique_ptr<Layer>& layer : layers) {
        activation = layer->feedforward(activation, evaluation_mode);
    }

    return activation;
}

void NeuralNetwork::backprop(const xt::xarray<float>& inputs, const xt::xarray<float>& labels) {
    xt::xarray<float> activation = feedforward(inputs, false);
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
                                float lr, float beta1, float beta2, float epsilon) {
    backprop(inputs, labels);

    for (std::unique_ptr<Layer>& layer : layers) {
        layer->update_adam(lr, beta1, beta2, epsilon);
    }
}

float NeuralNetwork::evaluate(const xt::xarray<float>& inputs, const xt::xarray<float>& labels) {
    size_t batch_size = inputs.shape()[0];

    xt::xarray<float> activations = feedforward(inputs, true);

    xt::xarray<size_t> pred_indices = xt::argmax(activations, 1);
    xt::xarray<size_t> true_indices = xt::argmax(labels, 1);

    int correct = 0;

    for (size_t i = 0; i < batch_size; i++) {
        if (pred_indices(i) == true_indices(i)) {
            correct++;
        }
    }

    return static_cast<float>(correct) / static_cast<float>(batch_size);
}

void NeuralNetwork::SGD(const std::vector<xt::xarray<float>>& training_inputs,
                        const std::vector<xt::xarray<float>>& training_labels,
                        const std::vector<xt::xarray<float>>& test_inputs,
                        const std::vector<xt::xarray<float>>& test_labels,
                        size_t epochs, size_t mini_batch_size, float lr) {

    auto conv_test_inputs = convert_vec_inputs(test_inputs);
    auto conv_test_labels = convert_vec_inputs(test_labels);

    size_t n = training_inputs.size();

    std::random_device rd;
    std::mt19937 g(rd());

    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    for (int epoch = 0; epoch < epochs; epoch++) {
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

            auto elapsed_time = std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::high_resolution_clock::now() - start_time
            ).count();

            float avg_time_per_batch = static_cast<float>(elapsed_time) / static_cast<float>(batch_index + 1);
            int remaining_batches = static_cast<int>(num_batches - (batch_index + 1));
            int estimated_remaining_time = remaining_batches * static_cast<int>(avg_time_per_batch);

            int minutes = estimated_remaining_time / 60;
            int seconds = estimated_remaining_time % 60;

            float percent_complete = (static_cast<float>(batch_index + 1) /
                                       static_cast<float>(num_batches)) * 100;

            std::cout << "\rEpoch #" << (epoch + 1)
                      << " progress: " << std::fixed << std::setprecision(4) << percent_complete << "% complete | "
                      << "Estimated time remaining: " << minutes << ":" << (seconds < 10 ? "0" : "") << seconds;

            std::fflush(stdout);
        }

        float current_accuracy = evaluate(conv_test_inputs, conv_test_labels);

        std::cout << "\rEpoch #" << (epoch + 1) << " | Accuracy: " << current_accuracy << std::endl;
    }
}



void NeuralNetwork::Adam(const std::vector<xt::xarray<float>>& training_inputs,
                         const std::vector<xt::xarray<float>>& training_labels,
                         const std::vector<xt::xarray<float>>& test_inputs,
                         const std::vector<xt::xarray<float>>& test_labels,
                         size_t epochs, size_t mini_batch_size, float lr, float beta1, float beta2, float epsilon) {

    auto conv_test_inputs = convert_vec_inputs(test_inputs);
    auto conv_test_labels = convert_vec_inputs(test_labels);

    size_t n = training_inputs.size();

    std::random_device rd;
    std::mt19937 g(rd());

    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    for (int epoch = 0; epoch < epochs; epoch++) {
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

            update_adam(inputs, labels, lr, beta1, beta2, epsilon);

            auto elapsed_time = std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::high_resolution_clock::now() - start_time
            ).count();

            float avg_time_per_batch = static_cast<float>(elapsed_time) / static_cast<float>(batch_index + 1);
            int remaining_batches = static_cast<int>(num_batches - (batch_index + 1));
            int estimated_remaining_time = remaining_batches * static_cast<int>(avg_time_per_batch);

            int minutes = estimated_remaining_time / 60;
            int seconds = estimated_remaining_time % 60;

            float percent_complete = (static_cast<float>(batch_index + 1) /
                                      static_cast<float>(num_batches)) * 100;

            std::cout << "\rEpoch #" << (epoch + 1)
                      << " progress: " << std::fixed << std::setprecision(4) << percent_complete << "% complete | "
                      << "Estimated time remaining: " << minutes << ":" << (seconds < 10 ? "0" : "") << seconds;

            std::fflush(stdout);
        }

        float current_accuracy = evaluate(conv_test_inputs, conv_test_labels);

        std::cout << "\rEpoch #" << (epoch + 1) << " | Accuracy: " << current_accuracy << std::endl;
    }
}
