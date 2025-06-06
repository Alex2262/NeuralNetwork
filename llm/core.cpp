//
// Created by Alexander Tian on 4/15/25.
//

#include <iostream>
#include <fstream>

#include "core.h"

#include "../layers/attention.h"
#include "../layers/dense.h"
#include "../layers/dropout.h"
#include "../layers/embedding.h"
#include "../layers/flatten.h"
#include "../layers/normalize.h"
#include "../layers/projection.h"
#include "../layers/res_add.h"


std::vector<std::string> get_chars(std::string& s) {
    std::vector<std::string> ret;
    for (char c : s) {
        std::string char_str{c};
        ret.push_back(char_str);
    }

    return ret;
}

std::vector<std::string> get_lower_chars(std::string& s) {
    std::vector<std::string> ret;
    for (char c : s) {
        std::string char_str{static_cast<char>(std::tolower(c))};
        ret.push_back(char_str);
    }

    return ret;
}

std::vector<std::string> get_words(std::string& s) {
    std::vector<std::string> ret;

    std::string curr;
    for (char c : s) {
        if (isalpha(c)) {
            curr += c;
            continue;
        }

        if (c == ' ') {
            if (!curr.empty()) ret.push_back(curr);
            curr = ' ';
            continue;
        }

        if (c == '\'') {
            if (!curr.empty() && curr != " ") {
                ret.push_back(curr);
                curr = "";
            }
            curr += c;
            continue;
        }

        if (!curr.empty()) ret.push_back(curr);
        curr = "";
        curr += c;
        ret.push_back(curr);
        curr = "";
    }

    if (!curr.empty()) {
        ret.push_back(curr);
    }

    return ret;
}

std::vector<std::string> LLM::get_tokens(std::string& s) {
    if (llm_mode == LLM_Mode::CHARS) return get_chars(s);
    else if (llm_mode == LLM_Mode::LOWER_CHARS) return get_lower_chars(s);
    else return get_words(s);
}

void LLM::set_data() {
    for (std::string& file_name : file_names) {
        std::ifstream file(file_name);

        if (!file.is_open()) {
            throw std::runtime_error("Cannot Open File");
        }

        std::string line;
        size_t num_lines = 0;
        while (getline(file, line)) {
            line += "\n";

            std::vector<std::string> vec_inputs = get_tokens(line);

            for (std::string& s : vec_inputs) {
                if (encode_map.find(s) == encode_map.end()) {
                    std::cout << "ENCODING " << s << " as " << vocab_size << std::endl;
                    encode_map[s] = vocab_size;
                    decode_map[vocab_size] = s;
                    vocab_size++;
                }

                all_encoded.push_back(encode_map[s]);
            }

            num_lines++;
            if (num_lines >= MAX_LINES) break;
        }

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

    if (temperature == 0) {
        throw std::runtime_error("Error: Temperature = 0");
    }
}

LLM::LLM(size_t p_num_layers, size_t p_num_heads, size_t p_max_seq_len, size_t p_d_model, size_t p_k, float p_dropout_rate, float p_temperature, std::vector<std::string>& p_file_names)
         : nn({p_max_seq_len}, CostID::CEL) {

    num_heads     = p_num_heads;
    num_layers    = p_num_layers;
    max_seq_len   = p_max_seq_len;
    d_model       = p_d_model;
    dense_neurons = p_d_model * 4;
    vocab_size    = 0;
    k             = p_k;

    dropout_rate = p_dropout_rate;
    temperature  = p_temperature;

    training_data_size = 0;

    input_size = {max_seq_len};
    file_names = p_file_names;

    set_data();
    sanity_checks();

    nn.add_layer<Embedding>(vocab_size, d_model, ActivationID::NONE);
    auto* embedding_layer = dynamic_cast<Embedding*>(nn.get_layer(0));

    nn.add_layer<Dropout>(dropout_rate);

    for (size_t layer = 0; layer < num_layers; layer++) {
        Layer* prev_out = nn.get_layer(nn.get_num_layers() - 1);

        nn.add_layer<Normalize>();
        nn.add_layer<Attention>(num_heads, ActivationID::NONE);
        nn.add_layer<Dropout>(dropout_rate);

        nn.add_layer<ResAdd>(prev_out);

        Layer* res_out_1 = nn.get_layer(nn.get_num_layers() - 1);

        nn.add_layer<Normalize>();
        nn.add_layer<Dense>(dense_neurons, ActivationID::GELU);
        nn.add_layer<Dense>(d_model, ActivationID::NONE);  // linear transformation back to D, no activation
        nn.add_layer<Dropout>(dropout_rate);

        nn.add_layer<ResAdd>(res_out_1);
    }

    nn.add_layer<Normalize>();

    nn.add_layer<Projection>(embedding_layer, k, temperature, ActivationID::SOFTMAX);

    std::cout << "LLM Initialization Complete" << std::endl;
    std::cout << "# params: " << nn.get_num_params() << std::endl;
}


void LLM::split_encoded(float split) {
    training_data_size = 0;
    test_data_size = 0;

    auto amt = static_cast<size_t>(static_cast<float>(all_encoded.size()) * split);

    for (size_t i = 0; i < amt; i++) {
        training_data_size++;
        train_encoded.push_back(all_encoded[i]);
    }

    for (size_t i = amt; i < all_encoded.size(); i++) {
        test_data_size++;
        test_encoded.push_back(all_encoded[i]);
    }

    std::cout << "Training Data Size: " << training_data_size << std::endl;
    std::cout << "Test Data Size: " << test_data_size << std::endl;
}

std::pair<xt::xtensor<float, 2>, xt::xtensor<float, 3>> LLM::index_test_batch(size_t batch_size, size_t ind) {
    xt::xtensor<float, 2> data = xt::zeros<float>({batch_size, max_seq_len});
    xt::xtensor<float, 3> labels = xt::zeros<float>({batch_size, max_seq_len, vocab_size});

    for (size_t batch = 0; batch < batch_size; batch++) {
        size_t start = ind + batch * max_seq_len;
        for (size_t i = 0; i < max_seq_len; i++) {
            data(batch, i) = static_cast<float>(test_encoded[start + i]);
            labels(batch, i, test_encoded[start + i + 1]) = 1.0f;
        }
    }

    return {data, labels};
}

float LLM::batched_evaluate(size_t batch_size) {
    size_t num_batches = (test_encoded.size() - 1) / (batch_size * max_seq_len);

    float accuracy = 0;
    for (size_t batch = 0; batch < num_batches; batch++) {
        size_t ind = batch * batch_size * max_seq_len;
        auto [test_data, test_labels] = index_test_batch(batch_size, ind);

        accuracy += nn.evaluate(test_data, test_labels);

        float progress = static_cast<float>(batch + 1) / static_cast<float>(num_batches) * 100;

        std::cout << "\r Evaluation Progress: " << std::fixed << std::setprecision(4) << progress << "% complete";
        std::fflush(stdout);
    }

    accuracy /= static_cast<float>(num_batches);
    return accuracy;
}

float LLM::batched_loss(size_t batch_size) {
    size_t num_batches = (test_encoded.size() - 1) / (batch_size * max_seq_len);

    float loss = 0;
    for (size_t batch = 0; batch < num_batches; batch++) {
        size_t ind = batch * batch_size * max_seq_len;
        auto [test_data, test_labels] = index_test_batch(batch_size, ind);

        loss += nn.loss(test_data, test_labels);

        float progress = static_cast<float>(batch + 1) / static_cast<float>(num_batches) * 100;

        std::cout << "\r Loss Progress: " << std::fixed << std::setprecision(4) << progress << "% complete";
        std::fflush(stdout);
    }

    loss /= static_cast<float>(num_batches);
    return loss;
}


std::pair<xt::xtensor<float, 2>, xt::xtensor<float, 3>> LLM::get_random_batch(size_t batch_size) {
    std::uniform_int_distribution<size_t> dis(0, train_encoded.size() - max_seq_len - 1);

    xt::xtensor<float, 2> data = xt::zeros<float>({batch_size, max_seq_len});
    xt::xtensor<float, 3> labels = xt::zeros<float>({batch_size, max_seq_len, vocab_size});

    for (size_t batch = 0; batch < batch_size; batch++) {
        size_t start = dis(gen);
        for (size_t i = 0; i < max_seq_len; i++) {
            data(batch, i) = static_cast<float>(train_encoded[start + i]);
            labels(batch, i, train_encoded[start + i + 1]) = 1.0f;
        }
    }

    return {data, labels};
}


void LLM::train(TrainInfo p_train_info) {

    nn.set_train_info(p_train_info);
    TrainInfo* train_info = nn.get_train_info();

    size_t num_super_batch = train_info->num_super_batch;
    size_t super_batch_size = train_info->super_batch_size;
    size_t mini_batch_size = train_info->mini_batch_size;
    size_t start_super_batch = train_info->current_super_batch + 1;

    float base_lr = train_info->lr;
    float beta1 = train_info->beta1;
    float beta2 = train_info->beta2;
    float weight_decay = train_info->weight_decay;

    if (num_super_batch == 0 || super_batch_size == 0 || mini_batch_size == 0 || base_lr == 0 || beta1 == 0 || beta2 == 0 || weight_decay == 0) {
        throw std::runtime_error("Set LLM Train Info: num_super_batch | super_batch_size | mini_batch_size | lr | beta1 | beta2 | weight_decay");
    }

    gen.seed(rd());

    std::cout << "Projecting to train on " << num_super_batch * super_batch_size * mini_batch_size * max_seq_len << " tokens" << std::endl;

    size_t num_tokens = (start_super_batch - 1) * super_batch_size * mini_batch_size * max_seq_len;

    auto start_time = std::chrono::high_resolution_clock::now();
    float sum_eval_time = 0;

    size_t total_steps = num_super_batch * super_batch_size;

    for (size_t super_batch = start_super_batch; super_batch <= num_super_batch; super_batch++) {
        for (size_t mini_batch = 1; mini_batch <= super_batch_size; mini_batch++) {
            num_tokens += mini_batch_size * max_seq_len;

            auto [train_data, train_labels] = get_random_batch(mini_batch_size);

            size_t current_step = (super_batch - 1) * super_batch_size + mini_batch;
            float lr = get_lr_linear_warmup_cosine_decay(current_step, total_steps, base_lr);

            nn.update_adamw(train_data, train_labels, lr, beta1, beta2, weight_decay);

            auto curr = std::chrono::high_resolution_clock::now();

            float percent_complete           = static_cast<float>(mini_batch) / static_cast<float>(super_batch_size) * 100.0f;
            float elapsed_time               = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(curr - start_time).count()) / 1000.0f;
            float tot_mini_batch_time        = elapsed_time - sum_eval_time;
            float avg_mini_batch_time        = tot_mini_batch_time / static_cast<float>((super_batch - start_super_batch) * super_batch_size + mini_batch);
            float avg_token_time             = avg_mini_batch_time / static_cast<float>(mini_batch_size * max_seq_len);
            float avg_eval_time              = super_batch == start_super_batch ? (0.5f * avg_token_time * static_cast<float>(test_data_size)) : (sum_eval_time / static_cast<float>(super_batch - start_super_batch));
            float avg_super_batch_time       = avg_mini_batch_time * static_cast<float>(super_batch_size) + avg_eval_time;
            float remaining_super_batch_time = avg_super_batch_time - avg_mini_batch_time * static_cast<float>(mini_batch);
            float remaining_total_time       = avg_super_batch_time * static_cast<float>(num_super_batch - super_batch + 1) - remaining_super_batch_time;

            std::cout << "\rToken #" << num_tokens
                      << " | Super Batch: " << super_batch
                      << " | Progress: " << std::fixed << std::setprecision(4) << percent_complete << "% complete"
                      << " | Elapsed: "  << format_time(static_cast<size_t>(elapsed_time))
                      << " | Remaining Super Batch Time: " << format_time(static_cast<size_t>(remaining_super_batch_time))
                      << " | Remaining Total Time: " << format_time(static_cast<size_t>(remaining_total_time))
                      << " | Avg Super Batch Time: " << format_time(static_cast<size_t>(avg_super_batch_time))
                      << std::flush;
        }

        auto eval_start_time    = std::chrono::high_resolution_clock::now();

        float current_accuracy  = batched_evaluate(mini_batch_size);
        float current_loss      = batched_loss(mini_batch_size);

        auto eval_end_time      = std::chrono::high_resolution_clock::now();
        float eval_elapsed_time = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(eval_end_time - eval_start_time).count()) / 1000.0f;
        sum_eval_time += eval_elapsed_time;

        std::cout << "\rToken #" << num_tokens << " | Accuracy: " << current_accuracy << " | Loss: " << current_loss << std::endl;

        train_info->current_super_batch = super_batch;

        std::string save_prefix = train_info->save_prefix + "_sb_" + std::to_string(super_batch);
        nn.save(save_prefix);
    }
}


size_t LLM::sample(const xt::xtensor<float, 3>& activations, size_t batch, size_t idx) {
    gen.seed(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    float r = dist(gen);
    float sum = 0.0f;

    size_t best = 0;

    for (size_t i = 0; i < vocab_size; i++) {
        sum += activations(batch, idx, i);
        if (sum >= r) {
            best = i;
            break;
        }
    }

    return best;
}


void LLM::run() {
    std::string input;
    while (getline(std::cin, input)) {
        std::vector<std::string> vec_inputs = get_tokens(input);
        size_t curr_seq_len = vec_inputs.size();

        if (curr_seq_len >= max_seq_len) {
            std::cout << "EXCEEDED SEQUENCE LENGTH" << std::endl;
            continue;
        }

        xt::xtensor<float, 2> tensor_inputs = xt::zeros<float>(std::vector<size_t>{1, max_seq_len});

        for (int i = 0; i < curr_seq_len; i++) tensor_inputs(0, i) = static_cast<float>(encode_map[vec_inputs[i]]);

        while (curr_seq_len < max_seq_len) {

            xt::xtensor<float, 3> activations = nn.feedforward(tensor_inputs, Mode::INFERENCE);

            size_t best = sample(activations, 0, curr_seq_len - 1);

            tensor_inputs(0, curr_seq_len) = static_cast<float>(best);
            std::cout << decode_map[best] << std::flush;
            curr_seq_len++;
        }

        std::cout << std::endl;
    }
}


void LLM::gen_file(std::string file_name, size_t num_tokens) {
    std::ofstream file;

    file.open(file_name);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot Open File");
    }

    xt::xtensor<float, 2> tensor_inputs = xt::zeros<float>(std::vector<size_t>{1, max_seq_len});

    tensor_inputs(0, 0) = static_cast<float>(encode_map["\n"]);
    size_t curr_seq_len = 1;

    while (curr_seq_len < num_tokens) {
        xt::xtensor<float, 3> activations = nn.feedforward(tensor_inputs, Mode::INFERENCE);
        size_t idx = std::min(curr_seq_len, max_seq_len) - 1;

        size_t best = sample(activations, 0, idx);

        if (curr_seq_len >= max_seq_len) {
            for (int i = 0; i < max_seq_len - 1; i++) {
                tensor_inputs(0, i) = tensor_inputs(0, i + 1);
            }

            tensor_inputs(0, idx) = static_cast<float>(best);
        }

        else tensor_inputs(0, curr_seq_len) = static_cast<float>(best);

        curr_seq_len++;

        file << decode_map[best];

        float percent = static_cast<float>(curr_seq_len) / static_cast<float>(num_tokens) * 100;

        std::cout << "\rPercent Written: " << std::fixed << std::setprecision(4) << percent << std::flush;
    }

    file << std::endl;
    std::cout << std::endl;

    file.close();
}


void LLM::load(std::string& file_prefix) {
    nn.load(file_prefix);
}