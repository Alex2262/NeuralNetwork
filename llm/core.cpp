//
// Created by Alexander Tian on 4/15/25.
//

#include <iostream>
#include <fstream>

#include "core.h"

#include "../layers/attention.h"
#include "../layers/dense.h"
#include "../layers/embedding.h"
#include "../layers/flatten.h"
#include "../layers/normalize.h"
#include "../layers/projection.h"
#include "../layers/res_add.h"



void LLM::set_data() {
    for (std::string& file_name : file_names) {
        std::ifstream file(file_name);

        if (!file.is_open()) {
            throw std::runtime_error("Cannot Open File");
        }

        std::vector<size_t> file_data;

        std::string line;
        size_t num_lines = 0;
        while (getline(file, line)) {
            line += "\n";
            for (char c : line) {
                if (encode_map.find(c) == encode_map.end()) {
                    // std::cout << "ENCODING " << c << " to " << vocab_size << std::endl;
                    encode_map[c] = vocab_size;
                    decode_map[vocab_size] = c;
                    vocab_size++;
                }

                file_data.push_back(encode_map[c]);
            }

            num_lines++;
            if (num_lines >= MAX_LINES) break;
        }

        all_encoded.push_back(file_data);
        file.close();
    }

    std::cout << "# of files parsed: " << file_names.size() << std::endl;
    std::cout << "Vocab Size: " << vocab_size << std::endl;
}

void LLM::sanity_checks() const {
    if (d_model % num_heads != 0) {
        throw std::runtime_error("Error: d_model must be divisible by num_heads.");
    }

    if (vocab_size == 0) {
        throw std::runtime_error("Error: Vocab Size = 0");
    }
}

LLM::LLM(size_t p_num_layers, size_t p_num_heads, size_t p_max_seq_len, size_t p_d_model, size_t p_dense_neurons, std::vector<std::string>& p_file_names)
         : nn({p_max_seq_len}, CostID::CEL) {

    num_heads     = p_num_heads;
    num_layers    = p_num_layers;
    max_seq_len   = p_max_seq_len;
    d_model       = p_d_model;
    dense_neurons = p_dense_neurons;
    vocab_size    = 0;

    training_data_size = 0;

    input_size = {max_seq_len};
    file_names = p_file_names;

    set_data();
    sanity_checks();

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

    std::cout << "LLM Initialization Complete" << std::endl;
}


void LLM::split_encoded(float split) {
    training_data_size = 0;

    for (const auto& file_data : all_encoded) {
        train_encoded.emplace_back();
        test_encoded.emplace_back();

        auto amt = static_cast<size_t>(static_cast<float>(file_data.size()) * split);
        for (size_t i = 0; i < amt; i++) {
            training_data_size++;
            train_encoded.back().push_back(file_data[i]);
        }

        for (size_t i = amt; i < file_data.size(); i++) {
            test_encoded.back().push_back(file_data[i]);
        }
    }

    std::cout << "Training Data Size: " << training_data_size << std::endl;
}


std::pair<xt::xtensor<float, 2>, xt::xtensor<float, 3>> LLM::get_test_data() {
    size_t total_size = 0;
    for (const auto& file_data : test_encoded) total_size += (file_data.size() - max_seq_len);

    xt::xtensor<float, 2> test_data = xt::zeros<float>({total_size, max_seq_len});
    xt::xtensor<float, 3> test_label = xt::zeros<float>({total_size, max_seq_len, vocab_size});

    size_t ind = 0;
    for (const auto& file_data : test_encoded) {
        for (size_t start = 0; start <= file_data.size() - max_seq_len - 1; start++) {
            for (size_t i = 0; i < max_seq_len; i++) {
                test_data(ind, i) = static_cast<float>(file_data[start + i]);
                test_label(ind, i, file_data[start + i + 1]) = 1.0f;
            }

            ind++;
        }
    }

    return {test_data, test_label};
}


std::pair<xt::xtensor<float, 2>, xt::xtensor<float, 3>> LLM::get_random_batch(size_t batch_size) {
    std::vector<std::uniform_int_distribution<size_t>> dis;
    dis.reserve(train_encoded.size());

    for (const auto& file_data : train_encoded) {
        dis.emplace_back(0, file_data.size() - max_seq_len - 1);
    }

    std::uniform_int_distribution<size_t> file_chooser_dis(0, training_data_size - 1);

    xt::xtensor<float, 2> data = xt::zeros<float>({batch_size, max_seq_len});
    xt::xtensor<float, 3> labels = xt::zeros<float>({batch_size, max_seq_len, vocab_size});

    for (size_t batch = 0; batch < batch_size; batch++) {
        size_t file_choice_rn = file_chooser_dis(gen);
        size_t file_choice = 0;
        size_t accum_size = 0;
        for (const auto& file_data : train_encoded) {
            accum_size += file_data.size();
            if (accum_size > file_choice_rn) break;
            file_choice++;
        }

        size_t start = dis[file_choice](gen);
        for (size_t i = 0; i < max_seq_len; i++) {
            data(batch, i) = static_cast<float>(train_encoded[file_choice][start + i]);
            labels(batch, i, train_encoded[file_choice][start + i + 1]) = 1.0f;
        }
    }

    return {data, labels};
}


void LLM::train(size_t epochs, size_t mini_batch_size, float split, float lr, float beta1, float beta2, float epsilon) {
    gen.seed(rd());

    split_encoded(split);
    auto [test_data, test_labels] = get_test_data();

    for (size_t epoch = 1; epoch <= epochs; epoch++) {
        size_t num_batches = (training_data_size + mini_batch_size - 1) / mini_batch_size;

        for (size_t batch = 1; batch <= num_batches; batch++) {
            auto [train_data, train_labels] = get_random_batch(mini_batch_size);

            /*
            std::cout << "BATCH DATA RECEIVED" << std::endl;

            for (size_t b = 0; b < mini_batch_size; b++) {
                std::string s;
                for (size_t i = 0; i < max_seq_len; i++) {
                    s += decode_map[static_cast<size_t>(train_data(b, i))];
                }

                std::cout << s << std::endl;
            }
             */


            nn.update_adam(train_data, train_labels, lr, beta1, beta2, epsilon);
            float percent_complete = (static_cast<float>(batch) /
                                      static_cast<float>(num_batches)) * 100;

            std::cout << "\rEpoch #" << epoch << " progress: " << std::fixed << std::setprecision(4) << percent_complete << "% complete";
            std::fflush(stdout);
        }

        float current_accuracy = nn.evaluate(test_data, test_labels);
        float current_loss     = nn.loss(test_data, test_labels);
        std::cout << "\rEpoch #" << epoch << " | Accuracy: " << current_accuracy << " | Loss: " << current_loss << std::endl;
    }
}




void LLM::run() {
    std::string input;
    size_t curr_seq_len = 0;
    while (getline(std::cin, input)) {
        curr_seq_len = input.size();

        if (curr_seq_len >= max_seq_len) {
            std::cout << "EXCEEDED SEQUENCE LENGTH" << std::endl;
            continue;
        }

        xt::xtensor<float, 2> tensor_inputs = xt::zeros<float>(std::vector<size_t>{1, max_seq_len});

        for (int i = 0; i < curr_seq_len; i++) tensor_inputs(0, i) = static_cast<float>(encode_map[input[i]]);

        while (curr_seq_len < max_seq_len) {

            xt::xtensor<float, 3> activations = nn.feedforward(tensor_inputs, true);

            float best_prob = 0;
            size_t best = 0;
            for (int i = 0; i < vocab_size; i++) {
                if (activations(0, curr_seq_len - 1, i) >= best_prob) {
                    best_prob = activations(0, curr_seq_len - 1, i);
                    best = i;
                }
            }

            tensor_inputs(0, curr_seq_len) = static_cast<float>(best);
            std::cout << decode_map[best];
            curr_seq_len++;
        }

        std::cout << std::endl;
    }
}