
#include "test/mnist.h"
#include "llm/core.h"

extern "C" void openblas_set_num_threads(int num_threads);
extern "C" int openblas_get_num_threads();

int main() {
    openblas_set_num_threads(8);
    std::cout << "OpenBLAS threads: " << openblas_get_num_threads() << std::endl;

    // test_mnist_cnn();
    // test_mnist();

    std::vector<std::string> file_names = {"/Users/alexandertian/CLionProjects/NeuralNetwork/llm/data/tiny_shakespeare.txt"};
    LLM llm(4, 4, 64, 128, 512, file_names);
    llm.train(30, 64, 0.98, 0.001, 0.9, 0.95, 1e-9);
    llm.run();

    return 0;
}
