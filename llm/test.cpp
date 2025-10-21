//
// Created by Alexander Tian on 5/26/25.
//

#include "test.h"
#include "core.h"


void train_llm_names() {
    TrainInfo train_info;
    train_info.num_super_batch = 20;
    train_info.super_batch_size = 100;
    train_info.mini_batch_size = 32;
    train_info.lr = 0.0003;
    train_info.beta1 = 0.9;
    train_info.beta2 = 0.99;
    train_info.weight_decay = 0.01;
    train_info.save_prefix = "";

    std::string names_file_path = DIR_PATH + "NeuralNetwork/llm/data/names.txt";

    std::vector<std::string> file_names = {names_file_path};
    LLM llm(2, 2, 10, 128, 25, 0.2, 0.8, file_names);
    llm.split_encoded(0.9);

    std::string output_path = DIR_PATH + "NeuralNetwork/llm/outputs/test1.txt";

    llm.train(train_info);
    llm.gen_file(output_path, 1000);
    llm.run();
}


void train_llm() {
    size_t last_epoch = 0;

    TrainInfo train_info;
    train_info.num_super_batch = 30;
    train_info.super_batch_size = 100;
    train_info.mini_batch_size = 32;
    train_info.lr = 0.0003;
    train_info.beta1 = 0.9;
    train_info.beta2 = 0.99;
    train_info.weight_decay = 0.01;
    train_info.save_prefix = DIR_PATH + "NeuralNetwork/llm/saved/shakespeare_llm";

    std::string shakespeare_file_path = DIR_PATH + "NeuralNetwork/llm/data/tiny_shakespeare.txt";

    std::vector<std::string> file_names = {shakespeare_file_path};
    LLM llm(6, 6, 256, 384, 12, 0.2, 0.8, file_names);
    llm.split_encoded(0.95);

    // RESUME CODE
    if (last_epoch > 0) {
        std::string load_prefix = train_info.save_prefix + "_sb_" + std::to_string(last_epoch);
        llm.load(load_prefix);

        float current_accuracy = llm.batched_evaluate(train_info.mini_batch_size);
        float current_loss     = llm.batched_loss(train_info.mini_batch_size);
        std::cout << "\rAccuracy: " << current_accuracy << " | Loss: " << current_loss << std::endl;
    }
    // --

    std::string output_path = DIR_PATH + "NeuralNetwork/llm/outputs/test1.txt";

    llm.train(train_info);
    llm.gen_file(output_path, 20000);
    llm.run();
}