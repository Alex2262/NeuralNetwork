
#include "test/mnist.h"
#include "llm/core.h"

// extern "C" void openblas_set_num_threads(int num_threads);
// extern "C" int openblas_get_num_threads();

int main() {
    // openblas_set_num_threads(1);
    // std::cout << "OpenBLAS threads: " << openblas_get_num_threads() << std::endl;

    // test_mnist_cnn();
    // test_mnist();
    // demo();

    std::vector<std::string> file_names = {"/Users/alexandertian/CLionProjects/NeuralNetwork/llm/data/tiny_shakespeare.txt"};
    LLM llm(8, 6, 128, 384, 10, 0.0, 0.8, file_names);
    llm.train(2e6, 1e5, 32, 0.95, 0.0003, 0.9, 0.98, 0.01);
    llm.gen_file("/Users/alexandertian/CLionProjects/NeuralNetwork/llm/outputs/test1.txt", 1000);
    llm.run();

    return 0;
}
