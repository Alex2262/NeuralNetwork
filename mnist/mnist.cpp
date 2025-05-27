

#include <iostream>
#include <vector>
#include <random>
#include <opencv2/opencv.hpp>
#include <xtensor/containers/xarray.hpp>

#include "matio.h"
#include "mnist.h"

#include "../layers/activation.h"
#include "../layers/dense.h"
#include "../layers/convolution.h"
#include "../layers/flatten.h"
#include "../layers/max_pool.h"
#include "../layers/normalize.h"
#include "../layers/dropout.h"
#include "../layers/res_add.h"

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


void show_image(const xt::xarray<float>& image, int label, int pred, float prob) {
    cv::Mat img_mat(28, 28, CV_64F);
    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            img_mat.at<double>(i, j) = image(i * 28 + j);
        }
    }

    img_mat *= 255.0;
    img_mat.convertTo(img_mat, CV_8U);

    cv::resize(img_mat, img_mat, cv::Size(500, 500), 0, 0, cv::INTER_NEAREST);

    std::string label_str = "Label: " + std::to_string(label);
    std::string pred_str = "Pred: " + std::to_string(pred);
    std::string prob_str = "Prob: " + std::to_string(prob);
    cv::putText(img_mat, label_str, cv::Point(5, 430), cv::FONT_HERSHEY_SIMPLEX,
                1, cv::Scalar(400), 2);
    cv::putText(img_mat, pred_str, cv::Point(5, 460), cv::FONT_HERSHEY_SIMPLEX,
                1, cv::Scalar(400), 2);
    cv::putText(img_mat, prob_str, cv::Point(5, 490), cv::FONT_HERSHEY_SIMPLEX,
                1, cv::Scalar(400), 2);

    cv::imshow("MNIST Image", img_mat);
    cv::waitKey(0);
}


void demo() {
    std::vector<xt::xarray<float>> train_images, train_labels;
    std::vector<xt::xarray<float>> test_images, test_labels;

    load_mnist_data("/Users/alexandertian/CLionProjects/NeuralNetwork/test/mnist-original.mat",
                    train_images, train_labels, test_images, test_labels, 0.8, 70000);

    std::vector<size_t> input_size = {784};
    NeuralNetwork nn(input_size, CostID::CEL);

    nn.add_layer<Dense>(4096, ActivationID::RELU);
    nn.add_layer<Dense>(10, ActivationID::SOFTMAX);

    TrainInfo train_info;
    train_info.num_epochs = 5;
    train_info.mini_batch_size = 64;
    train_info.lr = 0.1;

    nn.set_train_info(train_info);
    nn.SGD(train_images, train_labels, test_images, test_labels);
    show_images(nn, train_images, train_labels);
}

void test_mnist() {
    std::vector<xt::xarray<float>> train_images, train_labels;
    std::vector<xt::xarray<float>> test_images, test_labels;

    load_mnist_data("/Users/alexandertian/CLionProjects/NeuralNetwork/test/mnist-original.mat",
                    train_images, train_labels, test_images, test_labels, 0.8, 70000);

    // ~96.7% accuracy
    std::vector<size_t> input_size = {784};
    NeuralNetwork nn(input_size, CostID::CEL);

    nn.add_layer<Dense>(256, ActivationID::NONE);
    nn.add_layer<Normalize>();
    nn.add_layer<Activation>(ActivationID::RELU);
    nn.add_layer<Dense>(10, ActivationID::SOFTMAX);

    /*
    nn.add_layer<Activation>(ActivationID::NONE);
    Layer* layer1 = nn.get_layer(nn.get_num_layers() - 1);

    nn.add_layer<Dense>(784, ActivationID::NONE);
    nn.add_layer<Normalize>();
    nn.add_layer<Activation>(ActivationID::RELU);
    Layer* layer2 = nn.get_layer(nn.get_num_layers() - 1);

    nn.add_layer<ResAdd>(layer1);

    nn.add_layer<Dense>(784, ActivationID::NONE);
    nn.add_layer<Normalize>();
    nn.add_layer<Activation>(ActivationID::RELU);

    nn.add_layer<ResAdd>(layer1);
    nn.add_layer<ResAdd>(layer2);

    nn.add_layer<Dense>(256, ActivationID::NONE);
    nn.add_layer<Normalize>();
    nn.add_layer<Activation>(ActivationID::RELU);

    nn.add_layer<Dense>(10, ActivationID::SOFTMAX);
     */

    // nn.SGD(train_images, train_labels, test_images, test_labels, 6, 64, 0.1);

    TrainInfo train_info;
    train_info.num_epochs = 10;
    train_info.mini_batch_size = 64;
    train_info.lr = 0.001;
    train_info.beta1 = 0.9;
    train_info.beta2 = 0.999;
    train_info.weight_decay = 0.01;
    train_info.save_prefix = "/Users/alexandertian/CLionProjects/NeuralNetwork/saved/mnist_nn";

    nn.set_train_info(train_info);
    nn.AdamW(train_images, train_labels, test_images, test_labels);
    show_images(nn, train_images, train_labels);
}


void test_load() {
    std::vector<xt::xarray<float>> train_images, train_labels;
    std::vector<xt::xarray<float>> test_images, test_labels;

    load_mnist_data("/Users/alexandertian/CLionProjects/NeuralNetwork/test/mnist-original.mat",
                    train_images, train_labels, test_images, test_labels, 0.8, 70000);

    auto conv_test_inputs = convert_vec_inputs(test_images);
    auto conv_test_labels = convert_vec_inputs(test_labels);

    std::vector<size_t> input_size = {784};
    NeuralNetwork nn(input_size, CostID::CEL);

    nn.add_layer<Dense>(30, ActivationID::NONE);
    nn.add_layer<Normalize>();
    nn.add_layer<Activation>(ActivationID::RELU);
    nn.add_layer<Dense>(10, ActivationID::SOFTMAX);

    TrainInfo train_info;
    train_info.num_epochs = 10;
    train_info.mini_batch_size = 64;
    train_info.lr = 0.001;
    train_info.beta1 = 0.9;
    train_info.beta2 = 0.999;
    train_info.weight_decay = 0.01;
    train_info.save_prefix = "/Users/alexandertian/CLionProjects/NeuralNetwork/saved/mnist_nn";

    nn.set_train_info(train_info);
    // nn.AdamW(train_images, train_labels, test_images, test_labels);

    std::string load_prefix = train_info.save_prefix + "_epoch10";

    nn.load(load_prefix);
    float current_accuracy = nn.evaluate(conv_test_inputs, conv_test_labels);

    std::cout << "Accuracy: " << current_accuracy << std::endl;
    show_images(nn, train_images, train_labels);
}

void test_resume_training() {
    std::vector<xt::xarray<float>> train_images, train_labels;
    std::vector<xt::xarray<float>> test_images, test_labels;

    load_mnist_data("/Users/alexandertian/CLionProjects/NeuralNetwork/test/mnist-original.mat",
                    train_images, train_labels, test_images, test_labels, 0.8, 70000);

    auto conv_test_inputs = convert_vec_inputs(test_images);
    auto conv_test_labels = convert_vec_inputs(test_labels);

    std::vector<size_t> input_size = {784};
    NeuralNetwork nn(input_size, CostID::CEL);

    nn.add_layer<Dense>(1024, ActivationID::NONE);
    nn.add_layer<Normalize>();
    nn.add_layer<Activation>(ActivationID::RELU);
    nn.add_layer<Dense>(10, ActivationID::SOFTMAX);

    TrainInfo train_info;
    train_info.num_epochs = 5;
    train_info.mini_batch_size = 64;
    train_info.lr = 0.001;
    train_info.beta1 = 0.9;
    train_info.beta2 = 0.999;
    train_info.weight_decay = 0.01;
    train_info.save_prefix = "/Users/alexandertian/CLionProjects/NeuralNetwork/mnist/saved/mnist_nn";

    nn.set_train_info(train_info);
    nn.AdamW(train_images, train_labels, test_images, test_labels);

    std::string load_prefix = train_info.save_prefix + "_epoch_5";

    nn.load(load_prefix);
    float current_accuracy = nn.evaluate(conv_test_inputs, conv_test_labels);

    train_info.num_epochs = 10;
    nn.set_train_info(train_info);
    nn.AdamW(train_images, train_labels, test_images, test_labels);

    std::cout << "Accuracy: " << current_accuracy << std::endl;
    show_images(nn, train_images, train_labels);
}


void test_mnist_cnn() {
    std::vector<xt::xarray<float>> train_images, train_labels;
    std::vector<xt::xarray<float>> test_images, test_labels;

    load_mnist_data("/Users/alexandertian/CLionProjects/NeuralNetwork/test/mnist-original.mat",
                    train_images, train_labels, test_images, test_labels, 0.8, 70000);

    train_images = get_3d(train_images);
    test_images = get_3d(test_images);

    std::vector<size_t> input_size = {28, 28, 1};
    NeuralNetwork nn(input_size, CostID::CEL);

    /*
    nn.add_layer<Convolution>(64, 5, 1, ActivationID::RELU);
    nn.add_layer<MaxPool>(2, 2);
    nn.add_layer<Flatten>();
    nn.add_layer<Dense>(1024, ActivationID::RELU);
    // nn.add_layer<Dropout>(0.25);
    nn.add_layer<Dense>(256, ActivationID::RELU);
    nn.add_layer<Dense>(10, ActivationID::SOFTMAX);
     */

    nn.add_layer<Convolution>(64, 5, 1, ActivationID::RELU);
    nn.add_layer<MaxPool>(2, 2);
    nn.add_layer<Flatten>();
    nn.add_layer<Dense>(256, ActivationID::NONE);
    nn.add_layer<Normalize>();
    nn.add_layer<Activation>(ActivationID::RELU);
    nn.add_layer<Dense>(10, ActivationID::SOFTMAX);

    // 97.8% accuracy with Convolution(64, 5, 1 RELU) --> Maxpool(2, 2) --> Flatten --> Dense(256, RELU) --> Dense(10, SOFTMAX);

    // nn.SGD(train_images, train_labels, test_images, test_labels, 10, 64, 0.02);

    TrainInfo train_info;
    train_info.num_epochs = 10;
    train_info.mini_batch_size = 64;
    train_info.lr = 0.001;
    train_info.beta1 = 0.9;
    train_info.beta2 = 0.999;
    train_info.weight_decay = 0.01;

    nn.set_train_info(train_info);
    nn.AdamW(train_images, train_labels, test_images, test_labels);
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

        xt::xarray<float> pred = nn.feedforward(sample, Mode::INFERENCE);

        int best = 0;
        float mx = 0;
        for (int i = 0; i < 10; i++) {
            if (pred(0, i) > mx) {
                best = i;
                mx = pred(0, i);
            }
            // text += "Digit " + std::to_string(i) + " Probability: " + std::to_string(pred(0, i)) + "\n";
        }

        show_image(train_images[ind], true_lab, best, mx);
        ind++;
    }
}
