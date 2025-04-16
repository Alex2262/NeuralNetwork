//
// Created by Alexander Tian on 4/15/25.
//

#ifndef NEURALNETWORK_CORE_H
#define NEURALNETWORK_CORE_H


#include "../neural_network.h"

class LLM {

private:
    std::vector<size_t> input_size;

    size_t num_heads;
    size_t num_layers;
    size_t max_seq_len;
    size_t d_model;
    size_t vocab_size;
    size_t dense_neurons;

    NeuralNetwork nn;

public:
    LLM(size_t p_num_heads, size_t p_num_layers, size_t p_max_seq_len, size_t p_vocab_size, size_t p_d_model, size_t p_dense_neurons);
};

#endif //NEURALNETWORK_CORE_H
