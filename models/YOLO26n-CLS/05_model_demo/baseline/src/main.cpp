#include "logger.h"
#include "preprocess.h"

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#define checkCuda(op) __check_cuda_runtime((op), #op, __FILE__, __LINE__)

bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line) {
    if (code != cudaSuccess) {
        std::fprintf(stderr, "%s:%d  %s failed.\n  code = %s, message = %s\n",
                     file, line, op, cudaGetErrorName(code), cudaGetErrorString(code));
        return false;
    }
    return true;
}

void test_cuda_preprocess() {
    const std::string engine_path = "models/yolo26n-cls_fp16.engine";
    const std::string image_path = "assets/sample.jpg";
    constexpr int resize_size = 224;
    constexpr int crop_size = 224;
    constexpr size_t input_elements = 3 * crop_size * crop_size;

    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("无法读取测试图片：" + image_path);
    }
    if (!image.isContinuous()) {
        image = image.clone();
    }
    const double scale = static_cast<double>(resize_size) / std::min(image.rows, image.cols);
    const int resized_width = static_cast<int>(image.cols * scale);
    const int resized_height = static_cast<int>(image.rows * scale);

    std::vector<std::string> labels;
    std::ifstream labels_file("assets/imagenet_classes.txt");
    if (!labels_file) {
        throw std::runtime_error("无法打开 ImageNet 类别文件");
    }
    std::string label;
    while (std::getline(labels_file, label)) {
        if (!label.empty() && label.back() == '\r') {
            label.pop_back();
        }
        labels.push_back(label);
    }
    if (labels.size() != 1000) {
        throw std::runtime_error("ImageNet 类别文件应包含 1000 行");
    }

    demo::Logger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(logger)};
    if (!runtime) {
        throw std::runtime_error("创建 TensorRT runtime 失败");
    }
    std::ifstream engine_file(engine_path, std::ios::binary | std::ios::ate);
    if (!engine_file) {
        throw std::runtime_error("无法打开 TensorRT engine：" + engine_path);
    }
    const std::streamsize engine_size = engine_file.tellg();
    if (engine_size <= 0) {
        throw std::runtime_error("TensorRT engine 文件为空");
    }
    std::vector<char> engine_data(static_cast<size_t>(engine_size));
    engine_file.seekg(0, std::ios::beg);
    if (!engine_file.read(engine_data.data(), engine_size)) {
        throw std::runtime_error("读取 TensorRT engine 失败");
    }
    std::unique_ptr<nvinfer1::ICudaEngine> engine{
        runtime->deserializeCudaEngine(engine_data.data(), engine_data.size())
    };
    if (!engine) {
        throw std::runtime_error("反序列化 TensorRT engine 失败");
    }
    std::unique_ptr<nvinfer1::IExecutionContext> context{engine->createExecutionContext()};
    if (!context) {
        throw std::runtime_error("创建 TensorRT execution context 失败");
    }

    const char* input_name = nullptr;
    const char* output_name = nullptr;
    for (int index = 0; index < engine->getNbIOTensors(); ++index) {
        const char* name = engine->getIOTensorName(index);
        if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            input_name = name;
        } else {
            output_name = name;
        }
    }
    if (input_name == nullptr || output_name == nullptr) {
        throw std::runtime_error("预期 engine 恰有一个输入 tensor 和一个输出 tensor");
    }
    const nvinfer1::Dims input_dims = engine->getTensorShape(input_name);
    const nvinfer1::Dims output_dims = engine->getTensorShape(output_name);
    if (input_dims.nbDims != 4 || input_dims.d[0] != 1 || input_dims.d[1] != 3 ||
        input_dims.d[2] != crop_size || input_dims.d[3] != crop_size ||
        output_dims.nbDims != 2 || output_dims.d[0] != 1 || output_dims.d[1] != 1000) {
        throw std::runtime_error("engine shape 不匹配；需要输入 [1,3,224,224] 和输出 [1,1000]");
    }

    cudaStream_t stream = nullptr;
    unsigned char* device_image = nullptr;
    float* device_input = nullptr;
    float* device_output = nullptr;
    if (!checkCuda(cudaStreamCreate(&stream)) ||
        !checkCuda(cudaMalloc(reinterpret_cast<void**>(&device_image), image.total() * image.elemSize())) ||
        !checkCuda(cudaMalloc(reinterpret_cast<void**>(&device_input), input_elements * sizeof(float))) ||
        !checkCuda(cudaMalloc(reinterpret_cast<void**>(&device_output), labels.size() * sizeof(float)))) {
        throw std::runtime_error("CUDA buffer 初始化失败");
    }
    if (!context->setTensorAddress(input_name, device_input) ||
        !context->setTensorAddress(output_name, device_output)) {
        throw std::runtime_error("TensorRT 输入 / 输出地址绑定失败");
    }

    // The CUDA path uploads the decoded BGR image and writes normalized NCHW directly to TensorRT input.
    auto preprocess = [&]() {
        if (!checkCuda(cudaMemcpyAsync(
            device_image, image.data, image.total() * image.elemSize(), cudaMemcpyHostToDevice, stream
        ))) {
            throw std::runtime_error("CUDA Runtime API call failed");
        }
        if (!checkCuda(launch_preprocess_kernel(
            device_image,
            device_input,
            image.cols,
            image.rows,
            resized_width,
            resized_height,
            (resized_width - crop_size) / 2,
            (resized_height - crop_size) / 2,
            stream
        ))) {
            throw std::runtime_error("CUDA preprocess kernel launch failed");
        }
        if (!checkCuda(cudaStreamSynchronize(stream))) {
            throw std::runtime_error("CUDA Runtime API call failed");
        }
    };

    auto preprocess_start = std::chrono::steady_clock::now();
    preprocess();
    const double preprocess_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - preprocess_start
    ).count();

    constexpr int warmup_iterations = 10;
    constexpr int timed_iterations = 100;
    for (int iteration = 0; iteration < warmup_iterations; ++iteration) {
        if (!context->enqueueV3(stream)) {
            throw std::runtime_error("TensorRT warmup inference 失败");
        }
    }
    if (!checkCuda(cudaStreamSynchronize(stream))) {
        throw std::runtime_error("CUDA Runtime API call failed");
    }
    const auto inference_start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < timed_iterations; ++iteration) {
        if (!context->enqueueV3(stream)) {
            throw std::runtime_error("TensorRT inference 失败");
        }
    }
    if (!checkCuda(cudaStreamSynchronize(stream))) {
        throw std::runtime_error("CUDA Runtime API call failed");
    }
    const double inference_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - inference_start
    ).count() / timed_iterations;

    // Measure one warmed request from preprocessing through the Top-1 result, excluding decode and engine setup.
    const auto end_to_end_start = std::chrono::steady_clock::now();
    preprocess();
    if (!context->enqueueV3(stream)) {
        throw std::runtime_error("TensorRT end-to-end inference 失败");
    }
    std::vector<float> output(labels.size());
    if (!checkCuda(cudaMemcpyAsync(
        output.data(), device_output, output.size() * sizeof(float), cudaMemcpyDeviceToHost, stream
    ))) {
        throw std::runtime_error("CUDA Runtime API call failed");
    }
    if (!checkCuda(cudaStreamSynchronize(stream))) {
        throw std::runtime_error("CUDA Runtime API call failed");
    }
    const int top1 = static_cast<int>(std::max_element(output.begin(), output.end()) - output.begin());
    const double confidence = 100.0 * output[top1];
    const double end_to_end_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - end_to_end_start
    ).count();

    std::cout << "[CUDA preprocessing + TensorRT]\n";
    std::cout << "Result: " << labels[top1] << " (" << std::fixed << std::setprecision(2)
              << confidence << "%)\n";
    std::cout << "Preprocess: " << std::setprecision(3) << preprocess_ms << " ms\n";
    std::cout << "Inference: " << inference_ms << " ms (100-run average)\n";
    std::cout << "End-to-end: " << end_to_end_ms << " ms\n\n";

    const bool sync_ok = checkCuda(cudaStreamSynchronize(stream));
    const bool image_free_ok = checkCuda(cudaFree(device_image));
    const bool input_free_ok = checkCuda(cudaFree(device_input));
    const bool output_free_ok = checkCuda(cudaFree(device_output));
    const bool stream_destroy_ok = checkCuda(cudaStreamDestroy(stream));
    if (!sync_ok || !image_free_ok || !input_free_ok || !output_free_ok || !stream_destroy_ok) {
        throw std::runtime_error("CUDA Runtime API cleanup 失败");
    }
}

void test_opencv_preprocess() {
    const std::string engine_path = "models/yolo26n-cls_fp16.engine";
    const std::string image_path = "assets/sample.jpg";
    constexpr int resize_size = 224;
    constexpr int crop_size = 224;
    constexpr size_t input_elements = 3 * crop_size * crop_size;

    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("无法读取测试图片：" + image_path);
    }
    const double scale = static_cast<double>(resize_size) / std::min(image.rows, image.cols);
    const int resized_width = static_cast<int>(image.cols * scale);
    const int resized_height = static_cast<int>(image.rows * scale);

    std::vector<std::string> labels;
    std::ifstream labels_file("assets/imagenet_classes.txt");
    if (!labels_file) {
        throw std::runtime_error("无法打开 ImageNet 类别文件");
    }
    std::string label;
    while (std::getline(labels_file, label)) {
        if (!label.empty() && label.back() == '\r') {
            label.pop_back();
        }
        labels.push_back(label);
    }
    if (labels.size() != 1000) {
        throw std::runtime_error("ImageNet 类别文件应包含 1000 行");
    }

    demo::Logger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(logger)};
    if (!runtime) {
        throw std::runtime_error("创建 TensorRT runtime 失败");
    }
    std::ifstream engine_file(engine_path, std::ios::binary | std::ios::ate);
    if (!engine_file) {
        throw std::runtime_error("无法打开 TensorRT engine：" + engine_path);
    }
    const std::streamsize engine_size = engine_file.tellg();
    if (engine_size <= 0) {
        throw std::runtime_error("TensorRT engine 文件为空");
    }
    std::vector<char> engine_data(static_cast<size_t>(engine_size));
    engine_file.seekg(0, std::ios::beg);
    if (!engine_file.read(engine_data.data(), engine_size)) {
        throw std::runtime_error("读取 TensorRT engine 失败");
    }
    std::unique_ptr<nvinfer1::ICudaEngine> engine{
        runtime->deserializeCudaEngine(engine_data.data(), engine_data.size())
    };
    if (!engine) {
        throw std::runtime_error("反序列化 TensorRT engine 失败");
    }
    std::unique_ptr<nvinfer1::IExecutionContext> context{engine->createExecutionContext()};
    if (!context) {
        throw std::runtime_error("创建 TensorRT execution context 失败");
    }

    const char* input_name = nullptr;
    const char* output_name = nullptr;
    for (int index = 0; index < engine->getNbIOTensors(); ++index) {
        const char* name = engine->getIOTensorName(index);
        if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            input_name = name;
        } else {
            output_name = name;
        }
    }
    if (input_name == nullptr || output_name == nullptr) {
        throw std::runtime_error("预期 engine 恰有一个输入 tensor 和一个输出 tensor");
    }
    const nvinfer1::Dims input_dims = engine->getTensorShape(input_name);
    const nvinfer1::Dims output_dims = engine->getTensorShape(output_name);
    if (input_dims.nbDims != 4 || input_dims.d[0] != 1 || input_dims.d[1] != 3 ||
        input_dims.d[2] != crop_size || input_dims.d[3] != crop_size ||
        output_dims.nbDims != 2 || output_dims.d[0] != 1 || output_dims.d[1] != 1000) {
        throw std::runtime_error("engine shape 不匹配；需要输入 [1,3,224,224] 和输出 [1,1000]");
    }

    cudaStream_t stream = nullptr;
    float* device_input = nullptr;
    float* device_output = nullptr;
    if (!checkCuda(cudaStreamCreate(&stream)) ||
        !checkCuda(cudaMalloc(reinterpret_cast<void**>(&device_input), input_elements * sizeof(float))) ||
        !checkCuda(cudaMalloc(reinterpret_cast<void**>(&device_output), labels.size() * sizeof(float)))) {
        throw std::runtime_error("CUDA buffer 初始化失败");
    }
    if (!context->setTensorAddress(input_name, device_input) ||
        !context->setTensorAddress(output_name, device_output)) {
        throw std::runtime_error("TensorRT 输入 / 输出地址绑定失败");
    }

    std::vector<float> opencv_input(input_elements);
    // This CPU preprocessing function also copies the final FP32 tensor to the TensorRT input buffer.
    auto preprocess = [&]() {
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(resized_width, resized_height), 0.0, 0.0, cv::INTER_LINEAR);
        const cv::Mat crop = resized(cv::Rect(
            (resized.cols - crop_size) / 2,
            (resized.rows - crop_size) / 2,
            crop_size,
            crop_size
        ));
        for (int y = 0; y < crop_size; ++y) {
            const cv::Vec3b* row = crop.ptr<cv::Vec3b>(y);
            for (int x = 0; x < crop_size; ++x) {
                for (int channel = 0; channel < 3; ++channel) {
                    opencv_input[(channel * crop_size + y) * crop_size + x] =
                        static_cast<float>(row[x][2 - channel]) / 255.0F;
                }
            }
        }
        if (!checkCuda(cudaMemcpyAsync(
            device_input, opencv_input.data(), input_elements * sizeof(float), cudaMemcpyHostToDevice, stream
        ))) {
            throw std::runtime_error("CUDA Runtime API call failed");
        }
        if (!checkCuda(cudaStreamSynchronize(stream))) {
            throw std::runtime_error("CUDA Runtime API call failed");
        }
    };

    auto preprocess_start = std::chrono::steady_clock::now();
    preprocess();
    const double preprocess_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - preprocess_start
    ).count();

    constexpr int warmup_iterations = 10;
    constexpr int timed_iterations = 100;
    for (int iteration = 0; iteration < warmup_iterations; ++iteration) {
        if (!context->enqueueV3(stream)) {
            throw std::runtime_error("TensorRT warmup inference 失败");
        }
    }
    if (!checkCuda(cudaStreamSynchronize(stream))) {
        throw std::runtime_error("CUDA Runtime API call failed");
    }
    const auto inference_start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < timed_iterations; ++iteration) {
        if (!context->enqueueV3(stream)) {
            throw std::runtime_error("TensorRT inference 失败");
        }
    }
    if (!checkCuda(cudaStreamSynchronize(stream))) {
        throw std::runtime_error("CUDA Runtime API call failed");
    }
    const double inference_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - inference_start
    ).count() / timed_iterations;

    // End-to-end is a warmed request from decoded image to the final Top-1 result.
    const auto end_to_end_start = std::chrono::steady_clock::now();
    preprocess();
    if (!context->enqueueV3(stream)) {
        throw std::runtime_error("TensorRT end-to-end inference 失败");
    }
    std::vector<float> output(labels.size());
    if (!checkCuda(cudaMemcpyAsync(
        output.data(), device_output, output.size() * sizeof(float), cudaMemcpyDeviceToHost, stream
    ))) {
        throw std::runtime_error("CUDA Runtime API call failed");
    }
    if (!checkCuda(cudaStreamSynchronize(stream))) {
        throw std::runtime_error("CUDA Runtime API call failed");
    }
    const int top1 = static_cast<int>(std::max_element(output.begin(), output.end()) - output.begin());
    const double confidence = 100.0 * output[top1];
    const double end_to_end_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - end_to_end_start
    ).count();

    std::cout << "[OpenCV preprocessing + TensorRT]\n";
    std::cout << "Result: " << labels[top1] << " (" << std::fixed << std::setprecision(2)
              << confidence << "%)\n";
    std::cout << "Preprocess: " << std::setprecision(3) << preprocess_ms << " ms\n";
    std::cout << "Inference: " << inference_ms << " ms (100-run average)\n";
    std::cout << "End-to-end: " << end_to_end_ms << " ms\n";

    const bool sync_ok = checkCuda(cudaStreamSynchronize(stream));
    const bool input_free_ok = checkCuda(cudaFree(device_input));
    const bool output_free_ok = checkCuda(cudaFree(device_output));
    const bool stream_destroy_ok = checkCuda(cudaStreamDestroy(stream));
    if (!sync_ok || !input_free_ok || !output_free_ok || !stream_destroy_ok) {
        throw std::runtime_error("CUDA Runtime API cleanup 失败");
    }
}

int main() {
    try {
        test_cuda_preprocess();
        test_opencv_preprocess();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}

#undef checkCuda
