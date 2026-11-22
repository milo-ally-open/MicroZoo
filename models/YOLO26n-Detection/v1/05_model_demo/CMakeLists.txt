cmake_minimum_required(VERSION 3.18)
project(yolo26n_demo_v1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h PATHS /usr/include/aarch64-linux-gnu /usr/include REQUIRED)
find_library(TENSORRT_LIBRARY nvinfer PATHS /usr/lib/aarch64-linux-gnu /usr/lib REQUIRED)

add_executable(yolo26n_demo src/main.cpp)
target_include_directories(yolo26n_demo PRIVATE include "${TENSORRT_INCLUDE_DIR}" ${OpenCV_INCLUDE_DIRS})
target_link_libraries(yolo26n_demo PRIVATE "${TENSORRT_LIBRARY}" CUDA::cudart ${OpenCV_LIBS})
