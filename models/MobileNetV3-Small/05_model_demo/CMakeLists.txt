cmake_minimum_required(VERSION 3.18)
project(mobilenet_v3_small_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    PATHS /usr/include/aarch64-linux-gnu /usr/include /usr/local/include
    REQUIRED
)
find_library(TENSORRT_LIBRARY nvinfer
    PATHS /usr/lib/aarch64-linux-gnu /lib/aarch64-linux-gnu /usr/local/lib
    REQUIRED
)

add_executable(mobilenet_demo src/main.cpp)
target_include_directories(mobilenet_demo PRIVATE include ${TENSORRT_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(mobilenet_demo PRIVATE ${TENSORRT_LIBRARY} CUDA::cudart ${OpenCV_LIBS})
target_compile_options(mobilenet_demo PRIVATE -Wall -Wextra -Wpedantic)
