

#ifndef NEURALNETWORK_MNIST_H
#define NEURALNETWORK_MNIST_H

#include <string>
#include <xtensor/containers/xarray.hpp>
#include "../neural_network.h"

void load_mnist_data(const std::string& file_path,
                     std::vector<xt::xarray<float>>& train_images,
                     std::vector<xt::xarray<float>>& train_labels,
                     std::vector<xt::xarray<float>>& test_images,
                     std::vector<xt::xarray<float>>& test_labels,
                     float train_ratio,
                     size_t total_amount);

std::vector<xt::xarray<float>> get_3d(std::vector<xt::xarray<float>>& images);

void show_image(const xt::xarray<float>& image, int label, int pred, float prob);

void show_images(NeuralNetwork& nn,
                 std::vector<xt::xarray<float>>& train_images,
                 std::vector<xt::xarray<float>>& train_labels);

void demo();
void test_mnist();
void test_mnist_cnn();

void test_load();
void test_resume_training();

#endif //NEURALNETWORK_MNIST_H
