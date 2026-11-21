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
#include <iterator>
#include <memory>
#include <numeric>
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

// Saved by the CUDA test so the OpenCV test can compare its normalized input tensor.
std::vector<float> cuda_preprocessed_tensor;

void test_cuda_preprocess() {
    const std::string engine_path = "models/mobilenet_v3_small_fp16.engine";
    const std::string image_path = "assets/dog.jpg";
    const std::string labels_path = "assets/imagenet_classes.txt";
    constexpr int resize_size = 256;
    constexpr int crop_size = 224;
    constexpr size_t input_elements = 3 * crop_size * crop_size;

    // This test owns its complete CUDA preprocessing and TensorRT inference pipeline.
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
    std::ifstream labels_file(labels_path);
    if (!labels_file) {
        throw std::runtime_error("无法打开 ImageNet 类别文件：" + labels_path);
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
    int input_count = 0;
    int output_count = 0;
    for (int index = 0; index < engine->getNbIOTensors(); ++index) {
        const char* name = engine->getIOTensorName(index);
        if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            input_name = name;
            ++input_count;
        } else {
            output_name = name;
            ++output_count;
        }
    }
    if (input_count != 1 || output_count != 1) {
        throw std::runtime_error("预期 engine 恰有一个输入 tensor 和一个输出 tensor");
    }
    const nvinfer1::Dims input_dims = engine->getTensorShape(input_name);
    const nvinfer1::Dims output_dims = engine->getTensorShape(output_name);
    if (input_dims.nbDims != 4 || input_dims.d[0] != 1 || input_dims.d[1] != 3 ||
        input_dims.d[2] != crop_size || input_dims.d[3] != crop_size ||
        output_dims.nbDims != 2 || output_dims.d[0] != 1 || output_dims.d[1] != 1000) {
        throw std::runtime_error("engine shape 不匹配；需要输入 [1,3,224,224] 和输出 [1,1000]");
    }
    if (engine->getTensorDataType(input_name) != nvinfer1::DataType::kFLOAT ||
        engine->getTensorDataType(output_name) != nvinfer1::DataType::kFLOAT ||
        engine->getTensorFormat(input_name) != nvinfer1::TensorFormat::kLINEAR ||
        engine->getTensorFormat(output_name) != nvinfer1::TensorFormat::kLINEAR) {
        throw std::runtime_error("此 demo 要求 FP32、LINEAR 格式的输入和输出 tensor");
    }

    cudaStream_t stream = nullptr;
    unsigned char* device_image = nullptr;
    float* device_input = nullptr;
    float* device_output = nullptr;
    if (!checkCuda(cudaStreamCreate(&stream))) {
        throw std::runtime_error("CUDA Runtime API call failed");
    }
    if (!checkCuda(cudaMalloc(reinterpret_cast<void**>(&device_image), image.total() * image.elemSize()))) {
        throw std::runtime_error("CUDA Runtime API call failed");
    }
    if (!checkCuda(cudaMalloc(reinterpret_cast<void**>(&device_input), input_elements * sizeof(float)))) {
        throw std::runtime_error("CUDA Runtime API call failed");
    }
    if (!checkCuda(cudaMalloc(reinterpret_cast<void**>(&device_output), labels.size() * sizeof(float)))) {
        throw std::runtime_error("CUDA Runtime API call failed");
    }
    if (!context->setTensorAddress(input_name, device_input) ||
        !context->setTensorAddress(output_name, device_output)) {
        throw std::runtime_error("TensorRT 输入 / 输出地址绑定失败");
    }

    // The timer covers raw-image H2D, the CUDA preprocessing kernel, and completion sync.
    const auto preprocess_start = std::chrono::steady_clock::now();
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
    const double preprocess_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - preprocess_start
    ).count();

    // Keep a host copy only for the comparison printed by the OpenCV test.
    cuda_preprocessed_tensor.resize(input_elements);
    if (!checkCuda(cudaMemcpyAsync(
        cuda_preprocessed_tensor.data(), device_input, input_elements * sizeof(float), cudaMemcpyDeviceToHost, stream
    ))) {
        throw std::runtime_error("CUDA Runtime API call failed");
    }
    if (!checkCuda(cudaStreamSynchronize(stream))) {
        throw std::runtime_error("CUDA Runtime API call failed");
    }

    // Warm up, then time repeated TensorRT inference calls with the CUDA-preprocessed input.
    constexpr int warmup_iterations = 10;
    constexpr int timed_iterations = 100;
    for (int iteration = 0; iteration < warmup_iterations; ++iteration) {
        if (!context->enqueueV3(stream)) {
            throw std::runtime_error("TensorRT warmup enqueueV3 推理失败");
        }
    }
    if (!checkCuda(cudaStreamSynchronize(stream))) {
        throw std::runtime_error("CUDA Runtime API call failed");
    }
    const auto inference_start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < timed_iterations; ++iteration) {
        if (!context->enqueueV3(stream)) {
            throw std::runtime_error("TensorRT enqueueV3 推理失败");
        }
    }
    if (!checkCuda(cudaStreamSynchronize(stream))) {
        throw std::runtime_error("CUDA Runtime API call failed");
    }
    const double inference_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - inference_start
    ).count() / timed_iterations;

    std::vector<float> output(labels.size());
    if (!checkCuda(cudaMemcpyAsync(
        output.data(), device_output, output.size() * sizeof(float), cudaMemcpyDeviceToHost, stream
    ))) {
        throw std::runtime_error("CUDA Runtime API call failed");
    }
    if (!checkCuda(cudaStreamSynchronize(stream))) {
        throw std::runtime_error("CUDA Runtime API call failed");
    }
    const float max_logit = *std::max_element(output.begin(), output.end());
    double exp_sum = 0.0;
    for (float logit : output) {
        exp_sum += std::exp(static_cast<double>(logit - max_logit));
    }
    std::vector<int> indices(output.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + 5, indices.end(), [&](int left, int right) {
        return output[left] > output[right];
    });

    std::cout << "=== CUDA preprocessing + TensorRT ===\n";
    std::cout << "Image: " << image_path << " (" << image.cols << 'x' << image.rows << ")\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Preprocess (H2D + CUDA kernel): " << preprocess_ms << " ms\n";
    std::cout << "TensorRT inference avg (" << timed_iterations << " runs): "
              << inference_ms << " ms/run (H2D/D2H excluded)\n";
    std::cout << "Top-5:\n";
    for (int rank = 0; rank < 5; ++rank) {
        const int index = indices[rank];
        const double probability = std::exp(static_cast<double>(output[index] - max_logit)) / exp_sum;
        std::cout << "  " << rank + 1 << ". " << labels[index]
                  << " (class " << index << ", " << probability * 100.0 << "%)\n";
    }

    const bool sync_ok = checkCuda(cudaStreamSynchronize(stream));
    const bool image_free_ok = checkCuda(cudaFree(device_image));
    const bool input_free_ok = checkCuda(cudaFree(device_input));
    const bool output_free_ok = checkCuda(cudaFree(device_output));
    const bool stream_destroy_ok = checkCuda(cudaStreamDestroy(stream));
    if (!sync_ok || !image_free_ok || !input_free_ok || !output_free_ok || !stream_destroy_ok) {
        throw std::runtime_error("CUDA Runtime API call failed");
    }
}

void test_opencv_preprocess() {
    const std::string engine_path = "models/mobilenet_v3_small_fp16.engine";
    const std::string image_path = "assets/dog.jpg";
    const std::string labels_path = "assets/imagenet_classes.txt";
    constexpr int resize_size = 256;
    constexpr int crop_size = 224;
    constexpr size_t input_elements = 3 * crop_size * crop_size;

    // This test independently runs CPU preprocessing followed by the same TensorRT engine.
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("无法读取测试图片：" + image_path);
    }
    const double scale = static_cast<double>(resize_size) / std::min(image.rows, image.cols);
    const int resized_width = static_cast<int>(image.cols * scale);
    const int resized_height = static_cast<int>(image.rows * scale);

    constexpr float mean[3] = {0.485F, 0.456F, 0.406F};
    constexpr float standard_deviation[3] = {0.229F, 0.224F, 0.225F};
    std::vector<float> opencv_input(input_elements);
    const auto preprocess_start = std::chrono::steady_clock::now();
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
                const float pixel = static_cast<float>(row[x][2 - channel]) / 255.0F;
                opencv_input[(channel * crop_size + y) * crop_size + x] =
                    (pixel - mean[channel]) / standard_deviation[channel];
            }
        }
    }
    const double opencv_preprocess_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - preprocess_start
    ).count();

    if (cuda_preprocessed_tensor.size() != input_elements) {
        throw std::runtime_error("请先运行 test_cuda_preprocess() 生成 CUDA 对照 tensor");
    }
    double max_abs_difference = 0.0;
    double total_abs_difference = 0.0;
    size_t max_difference_index = 0;
    size_t different_values = 0;
    for (size_t index = 0; index < input_elements; ++index) {
        const double difference = std::fabs(cuda_preprocessed_tensor[index] - opencv_input[index]);
        total_abs_difference += difference;
        if (difference > max_abs_difference) {
            max_abs_difference = difference;
            max_difference_index = index;
        }
        if (difference > 1.0e-3) {
            ++different_values;
        }
    }
    const double mean_abs_difference = total_abs_difference / input_elements;

    std::vector<std::string> labels;
    std::ifstream labels_file(labels_path);
    if (!labels_file) {
        throw std::runtime_error("无法打开 ImageNet 类别文件：" + labels_path);
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
    int input_count = 0;
    int output_count = 0;
    for (int index = 0; index < engine->getNbIOTensors(); ++index) {
        const char* name = engine->getIOTensorName(index);
        if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            input_name = name;
            ++input_count;
        } else {
            output_name = name;
            ++output_count;
        }
    }
    if (input_count != 1 || output_count != 1) {
        throw std::runtime_error("预期 engine 恰有一个输入 tensor 和一个输出 tensor");
    }
    const nvinfer1::Dims input_dims = engine->getTensorShape(input_name);
    const nvinfer1::Dims output_dims = engine->getTensorShape(output_name);
    if (input_dims.nbDims != 4 || input_dims.d[0] != 1 || input_dims.d[1] != 3 ||
        input_dims.d[2] != crop_size || input_dims.d[3] != crop_size ||
        output_dims.nbDims != 2 || output_dims.d[0] != 1 || output_dims.d[1] != 1000) {
        throw std::runtime_error("engine shape 不匹配；需要输入 [1,3,224,224] 和输出 [1,1000]");
    }
    if (engine->getTensorDataType(input_name) != nvinfer1::DataType::kFLOAT ||
        engine->getTensorDataType(output_name) != nvinfer1::DataType::kFLOAT ||
        engine->getTensorFormat(input_name) != nvinfer1::TensorFormat::kLINEAR ||
        engine->getTensorFormat(output_name) != nvinfer1::TensorFormat::kLINEAR) {
        throw std::runtime_error("此 demo 要求 FP32、LINEAR 格式的输入和输出 tensor");
    }

    cudaStream_t stream = nullptr;
    float* device_input = nullptr;
    float* device_output = nullptr;
    if (!checkCuda(cudaStreamCreate(&stream))) {
        throw std::runtime_error("CUDA Runtime API call failed");
    }
    if (!checkCuda(cudaMalloc(reinterpret_cast<void**>(&device_input), input_elements * sizeof(float)))) {
        throw std::runtime_error("CUDA Runtime API call failed");
    }
    if (!checkCuda(cudaMalloc(reinterpret_cast<void**>(&device_output), labels.size() * sizeof(float)))) {
        throw std::runtime_error("CUDA Runtime API call failed");
    }
    if (!context->setTensorAddress(input_name, device_input) ||
        !context->setTensorAddress(output_name, device_output)) {
        throw std::runtime_error("TensorRT 输入 / 输出地址绑定失败");
    }

    // Copy the CPU-preprocessed tensor to the device before inference; report it separately.
    const auto input_copy_start = std::chrono::steady_clock::now();
    if (!checkCuda(cudaMemcpyAsync(
        device_input, opencv_input.data(), input_elements * sizeof(float), cudaMemcpyHostToDevice, stream
    ))) {
        throw std::runtime_error("CUDA Runtime API call failed");
    }
    if (!checkCuda(cudaStreamSynchronize(stream))) {
        throw std::runtime_error("CUDA Runtime API call failed");
    }
    const double input_copy_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - input_copy_start
    ).count();

    // Warm up, then measure the same TensorRT engine using the OpenCV-preprocessed input.
    constexpr int warmup_iterations = 10;
    constexpr int timed_iterations = 100;
    for (int iteration = 0; iteration < warmup_iterations; ++iteration) {
        if (!context->enqueueV3(stream)) {
            throw std::runtime_error("TensorRT warmup enqueueV3 推理失败");
        }
    }
    if (!checkCuda(cudaStreamSynchronize(stream))) {
        throw std::runtime_error("CUDA Runtime API call failed");
    }
    const auto inference_start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < timed_iterations; ++iteration) {
        if (!context->enqueueV3(stream)) {
            throw std::runtime_error("TensorRT enqueueV3 推理失败");
        }
    }
    if (!checkCuda(cudaStreamSynchronize(stream))) {
        throw std::runtime_error("CUDA Runtime API call failed");
    }
    const double inference_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - inference_start
    ).count() / timed_iterations;

    std::vector<float> output(labels.size());
    if (!checkCuda(cudaMemcpyAsync(
        output.data(), device_output, output.size() * sizeof(float), cudaMemcpyDeviceToHost, stream
    ))) {
        throw std::runtime_error("CUDA Runtime API call failed");
    }
    if (!checkCuda(cudaStreamSynchronize(stream))) {
        throw std::runtime_error("CUDA Runtime API call failed");
    }
    const float max_logit = *std::max_element(output.begin(), output.end());
    double exp_sum = 0.0;
    for (float logit : output) {
        exp_sum += std::exp(static_cast<double>(logit - max_logit));
    }
    std::vector<int> indices(output.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + 5, indices.end(), [&](int left, int right) {
        return output[left] > output[right];
    });

    std::cout << "=== OpenCV preprocessing + TensorRT ===\n";
    std::cout << "Image: " << image_path << " (" << image.cols << 'x' << image.rows << ")\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Preprocess (OpenCV CPU): " << opencv_preprocess_ms << " ms\n";
    std::cout << "Input H2D copy: " << input_copy_ms << " ms\n";
    std::cout << "Input tensor difference vs CUDA: max_abs=" << max_abs_difference
              << " at index " << max_difference_index
              << ", mean_abs=" << mean_abs_difference
              << ", abs_diff > 1e-3: " << different_values << '/' << input_elements << '\n';
    std::cout << "First 8 NCHW values (CUDA | OpenCV):\n";
    for (size_t index = 0; index < 8; ++index) {
        std::cout << "  " << index << ": " << cuda_preprocessed_tensor[index]
                  << " | " << opencv_input[index] << '\n';
    }
    std::cout << "TensorRT inference avg (" << timed_iterations << " runs): "
              << inference_ms << " ms/run (H2D/D2H excluded)\n";
    std::cout << "Top-5:\n";
    for (int rank = 0; rank < 5; ++rank) {
        const int index = indices[rank];
        const double probability = std::exp(static_cast<double>(output[index] - max_logit)) / exp_sum;
        std::cout << "  " << rank + 1 << ". " << labels[index]
                  << " (class " << index << ", " << probability * 100.0 << "%)\n";
    }

    const bool sync_ok = checkCuda(cudaStreamSynchronize(stream));
    const bool input_free_ok = checkCuda(cudaFree(device_input));
    const bool output_free_ok = checkCuda(cudaFree(device_output));
    const bool stream_destroy_ok = checkCuda(cudaStreamDestroy(stream));
    if (!sync_ok || !input_free_ok || !output_free_ok || !stream_destroy_ok) {
        throw std::runtime_error("CUDA Runtime API call failed");
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
