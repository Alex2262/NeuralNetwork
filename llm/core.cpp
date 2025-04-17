//
// Created by Alexander Tian on 4/15/25.
//

#include "core.h"

#include "../layers/attention.h"
#include "../layers/dense.h"
#include "../layers/embedding.h"
#include "../layers/flatten.h"
#include "../layers/normalize.h"
#include "../layers/projection.h"
#include "../layers/res_add.h"


LLM::LLM(size_t p_num_heads, size_t p_num_layers, size_t p_max_seq_len, size_t p_vocab_size, size_t p_d_model, size_t p_dense_neurons)
         : nn({p_max_seq_len}, CostID::CEL) {

    num_heads     = p_num_heads;
    num_layers    = p_num_layers;
    max_seq_len   = p_max_seq_len;
    d_model       = p_d_model;
    vocab_size    = p_vocab_size;
    dense_neurons = p_dense_neurons;

    input_size = {max_seq_len};

    nn.add_layer<Embedding>(vocab_size, d_model, ActivationID::NONE);
    auto* embedding_layer = dynamic_cast<Embedding*>(nn.get_layer(0));

    for (size_t layer = 0; layer < num_layers; layer++) {
        Layer* prev_out = nn.get_layer(nn.get_num_layers() - 1);

        nn.add_layer<Normalize>();
        nn.add_layer<Attention>(num_heads, ActivationID::NONE);
        nn.add_layer<ResAdd>(prev_out);

        Layer* res_out_1 = nn.get_layer(nn.get_num_layers() - 1);

        nn.add_layer<Normalize>();
        nn.add_layer<Dense>(dense_neurons, ActivationID::RELU);
        nn.add_layer<Dense>(d_model, ActivationID::NONE);  // linear transformation back to D, no activation

        nn.add_layer<ResAdd>(res_out_1);
    }

    nn.add_layer<Normalize>();
    nn.add_layer<Projection>(embedding_layer, ActivationID::SOFTMAX);
}
