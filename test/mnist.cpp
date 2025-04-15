

#include <iostream>
#include <vector>
#include <random>
#include <opencv2/opencv.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xview.hpp>

#include "matio.h"
#include "mnist.h"

#include "../layers/dense.h"
#include "../layers/convolution.h"
#include "../layers/flatten.h"
#include "../layers/max_pool.h"

void load_mnist_data(const std::string& file_path,
                     std::vector<xt::xarray<float>>& train_images,
                     std::vector<xt::xarray<float>>& train_labels,
                     std::vector<xt::xarray<float>>& test_images,
                     std::vector<xt::xarray<float>>& test_labels,
                     float train_ratio,
                     size_t total_amount) {

    mat_t *mat_file_path = Mat_Open(file_path.c_str(), MAT_ACC_RDONLY);
    if (!mat_file_path) {
        std::cerr << "Error opening MAT file: " << file_path << std::endl;
        return;
    }

    matvar_t *images_var = Mat_VarRead(mat_file_path, "data");
    matvar_t *labels_var = Mat_VarRead(mat_file_path, "label");

    if (!images_var || !labels_var) {
        std::cerr << "Error reading dataset from MAT file." << std::endl;
        Mat_Close(mat_file_path);
        return;
    }

    size_t num_samples = images_var->dims[1];
    size_t image_size = images_var->dims[0];

    auto image_data = static_cast<uint8_t*>(images_var->data);
    auto label_data = static_cast<double*>(labels_var->data);

    std::vector<std::pair<xt::xarray<float>, xt::xarray<float>>> dataset;

    std::cout << "num_samples: " << num_samples << ", used samples: " << total_amount << ", image_size: " << image_size << std::endl;

    for (size_t i = 0; i < num_samples; ++i) {
        xt::xarray<float> image = xt::zeros<float>({image_size});
        xt::xarray<float> label = xt::zeros<float>({10});

        for (size_t j = 0; j < image_size; ++j) {
            image(j) = static_cast<float>(image_data[i * image_size + j]) / 255.0;
        }

        int label_index = static_cast<int>(label_data[i]);
        label(label_index) = 1.0;

        dataset.emplace_back(image, label);
    }

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(dataset.begin(), dataset.end(), g);

    total_amount = std::min<size_t>(total_amount, num_samples);

    auto train_size = static_cast<size_t>(static_cast<float>(total_amount) * train_ratio);

    for (size_t i = 0; i < total_amount; ++i) {
        if (i < train_size) {
            train_images.push_back(dataset[i].first);
            train_labels.push_back(dataset[i].second);
        } else {
            test_images.push_back(dataset[i].first);
            test_labels.push_back(dataset[i].second);
        }
    }

    Mat_VarFree(images_var);
    Mat_VarFree(labels_var);
    Mat_Close(mat_file_path);

    std::cout << "DATA LOADED" << std::endl;
}


std::vector<xt::xarray<float>> get_3d(std::vector<xt::xarray<float>>& images) {
    std::vector<xt::xarray<float>> converted;
    converted.reserve(images.size());

    for (xt::xarray<float>& image : images) {
        converted.push_back(xt::reshape_view(image, {28, 28, 1}));
    }

    return converted;
}


void show_image(const xt::xarray<float>& image) {
    cv::Mat img_mat(28, 28, CV_64F);
    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            img_mat.at<double>(i, j) = image(i * 28 + j);
        }
    }

    img_mat *= 255.0;
    img_mat.convertTo(img_mat, CV_8U);

    cv::imshow("MNIST Image", img_mat);
    cv::waitKey(0);
}


void test_mnist() {
    std::vector<xt::xarray<float>> train_images, train_labels;
    std::vector<xt::xarray<float>> test_images, test_labels;

    load_mnist_data("/Users/alexandertian/CLionProjects/NeuralNetwork/test/mnist-original.mat",
                    train_images, train_labels, test_images, test_labels, 0.8, 70000);

    std::vector<size_t> input_size = {784};
    NeuralNetwork nn(input_size, CostID::CEL);
    nn.add_layer<Dense>(128, ActivationID::RELU);
    nn.add_layer<Dense>(10, ActivationID::SOFTMAX);

    nn.SGD(train_images, train_labels, test_images, test_labels, 6, 64, 0.1);

    show_images(nn, train_images, train_labels);
}


void test_mnist_cnn() {
    std::vector<xt::xarray<float>> train_images, train_labels;
    std::vector<xt::xarray<float>> test_images, test_labels;

    load_mnist_data("/Users/alexandertian/CLionProjects/NeuralNetwork/test/mnist-original.mat",
                    train_images, train_labels, test_images, test_labels, 0.8, 20000);

    train_images = get_3d(train_images);
    test_images = get_3d(test_images);

    std::vector<size_t> input_size = {28, 28, 1};
    NeuralNetwork nn(input_size, CostID::CEL);
    nn.add_layer<Convolution>(64, 5, 1, ActivationID::RELU);
    nn.add_layer<MaxPool>(2, 2);
    nn.add_layer<Flatten>();
    nn.add_layer<Dense>(10, ActivationID::SOFTMAX);

    nn.SGD(train_images, train_labels, test_images, test_labels, 15, 64, 0.02);
}



void show_images(NeuralNetwork& nn,
                 std::vector<xt::xarray<float>>& train_images,
                 std::vector<xt::xarray<float>>& train_labels) {

    int ind = 0;

    while (true) {
        size_t true_lab = xt::argmax(train_labels[ind])();
        std::cout << "True Label: " << true_lab << std::endl;

        std::vector<xt::xarray<float>> vec_sample;
        vec_sample.push_back(train_images[ind]);

        xt::xarray<float> sample = convert_vec_inputs(vec_sample);

        xt::xarray<float> pred = nn.feedforward(sample, true);

        for (int i = 0; i < 10; i++) {
            std::cout << "Digit " << i << " Probability: " << pred(0, i) << std::endl;
        }

        show_image(train_images[ind]);
        ind++;
    }
}
