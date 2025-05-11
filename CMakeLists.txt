cmake_minimum_required(VERSION 3.28)
project(NeuralNetwork)

set(CMAKE_CXX_STANDARD 20)

add_definitions(-DHAVE_CBLAS=1)
add_definitions(-DXTENSOR_ENABLE_XSIMD=1)
set(XTENSOR_USE_XSIMD ON)

set(CMAKE_CXX_FLAGS_DEBUG "-Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-march=native -O3 -flto -ffast-math -funroll-loops")

find_library(MATIO_LIB matio HINTS /opt/homebrew/lib /usr/local/lib)
find_package(OpenCV REQUIRED)
find_package(xtensor CONFIG REQUIRED)
find_package(BLAS REQUIRED)

include_directories(/opt/homebrew/include)
include_directories(/Users/alexandertian/xtensor-blas/include)

add_executable(NeuralNetwork main.cpp
        layers/layers.h
        layers/dense.cpp
        layers/dense.h
        types.h
        utilities.cpp
        utilities.h
        neural_network.h
        neural_network.cpp
        test/mnist.h
        test/mnist.cpp
        layers/convolution.cpp
        layers/convolution.h
        layers/flatten.cpp
        layers/flatten.h
        layers/max_pool.cpp
        layers/max_pool.h
        layers/embedding.cpp
        layers/embedding.h
        layers/attention.cpp
        layers/attention.h
        layers/projection.cpp
        layers/projection.h
        llm/core.cpp
        llm/core.h
        layers/res_add.cpp
        layers/res_add.h
        layers/dropout.cpp
        layers/dropout.h
        layers/normalize.cpp
        layers/normalize.h
        layers/activation.cpp
        layers/activation.h
)

target_link_libraries(NeuralNetwork
        ${MATIO_LIB}
        ${OpenCV_LIBS}
        ${BLAS_LIBRARIES}
        xtensor
)