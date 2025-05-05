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

    LLM_MODE llm_mode = LLM_MODE::LOWER_CHARS;

    size_t MAX_LINES = 40000;

    size_t num_heads;
    size_t num_layers;
    size_t max_seq_len;
    size_t d_model;
    size_t dense_neurons;
    size_t vocab_size;
    size_t training_data_size;

    float dropout_rate;

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

    void split_encoded(float split);

    std::pair<xt::xtensor<float, 2>, xt::xtensor<float, 3>> index_test_batch(size_t batch_size, size_t ind);
    float batched_evaluate(size_t batch_size);
    float batched_loss(size_t batch_size);
    std::pair<xt::xtensor<float, 2>, xt::xtensor<float, 3>> get_random_batch(size_t batch_size);

public:
    LLM(size_t p_num_layers, size_t p_num_heads, size_t p_max_seq_len, size_t p_d_model, float p_dropout_rate, std::vector<std::string>& p_file_names);

    void train(size_t max_tokens, size_t eval_interval, size_t mini_batch_size, float split, float lr, float beta1, float beta2, float epsilon);

    void run();

};

#endif //NEURALNETWORK_CORE_H
