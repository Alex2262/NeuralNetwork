
#include "test/mnist.h"
#include "llm/core.h"

extern "C" void openblas_set_num_threads(int num_threads);
extern "C" int openblas_get_num_threads();

int main() {
    openblas_set_num_threads(8);
    std::cout << "OpenBLAS threads: " << openblas_get_num_threads() << std::endl;

    // test_mnist_cnn();
    // test_mnist();
    // demo();

    std::vector<std::string> file_names = {"/Users/alexandertian/CLionProjects/NeuralNetwork/llm/data/tiny_shakespeare.txt"};
    LLM llm(8, 6, 16, 384, 0.1, file_names);
    llm.train(6e6, 1e5, 32, 0.95, 0.0003, 0.9, 0.98, 1e-9);
    llm.run();

    // 1.96 loss:
    // LLM llm(8, 8, 64, 256, 0.15, file_names);
    // llm.train(2e6, 1e5, 32, 0.95, 0.0003, 0.9, 0.95, 1e-9);


    return 0;
}
