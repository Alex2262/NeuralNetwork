//
// Created by Alexander Tian on 4/15/25.
//

#ifndef NEURALNETWORK_CORE_H
#define NEURALNETWORK_CORE_H


#include <random>
#include "../neural_network.h"

std::vector<std::string> get_chars(std::string& s);
std::vector<std::string> get_words(std::string& s);



class LLM {

private:
    std::vector<size_t> input_size;

    LLM_Mode llm_mode = LLM_Mode::LOWER_CHARS;

    size_t MAX_LINES = 40000;

    size_t num_heads;
    size_t num_layers;
    size_t max_seq_len;
    size_t d_model;
    size_t dense_neurons;
    size_t vocab_size;
    size_t training_data_size;
    size_t test_data_size;
    size_t k;

    float dropout_rate;
    float temperature;

    NeuralNetwork nn;

    std::map<std::string, size_t> encode_map;
    std::map<size_t, std::string> decode_map;

    std::vector<std::string> file_names;
    std::vector<size_t> all_encoded;

    std::vector<size_t> train_encoded;
    std::vector<size_t> test_encoded;

    std::random_device rd;
    std::mt19937 gen;

    std::vector<std::string> get_tokens(std::string& s);

    void set_data();
    void sanity_checks() const;

    std::pair<xt::xtensor<float, 2>, xt::xtensor<float, 3>> index_test_batch(size_t batch_size, size_t ind);
    std::pair<xt::xtensor<float, 2>, xt::xtensor<float, 3>> get_random_batch(size_t batch_size);

    size_t sample(const xt::xtensor<float, 3>& activations, size_t batch, size_t idx);

public:
    LLM(size_t p_num_layers, size_t p_num_heads, size_t p_max_seq_len, size_t p_d_model, size_t p_k, float p_dropout_rate, float p_temperature, std::vector<std::string>& p_file_names);

    float batched_evaluate(size_t batch_size);
    float batched_loss(size_t batch_size);

    void split_encoded(float split);

    void train(TrainInfo p_train_info);

    void run();

    void gen_file(std::string file_name, size_t num_tokens);

    void load(std::string& file_prefix);

};

#endif //NEURALNETWORK_CORE_H
