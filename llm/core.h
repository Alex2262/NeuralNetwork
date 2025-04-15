//
// Created by Alexander Tian on 4/15/25.
//

#ifndef NEURALNETWORK_CORE_H
#define NEURALNETWORK_CORE_H


#include "../neural_network.h"

class LLM {

private:
    NeuralNetwork nn;
public:
    LLM(size_t p_num_heads, size_t p_num_layers, size_t d_model);
};

#endif //NEURALNETWORK_CORE_H
