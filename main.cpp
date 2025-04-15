
#include "test/mnist.h"

extern "C" void openblas_set_num_threads(int num_threads);
extern "C" int openblas_get_num_threads();

int main() {
    openblas_set_num_threads(8);
    std::cout << "OpenBLAS threads: " << openblas_get_num_threads() << std::endl;

    test_mnist_cnn();
    // test_mnist();

    return 0;
}
