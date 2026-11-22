#include "logger.h"

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK_CUDA(operation) \
    do { \
        const cudaError_t status = (operation); \
        if (status != cudaSuccess) { \
            throw std::runtime_error(cudaGetErrorString(status)); \
        } \
    } while (false)

int main() {
    constexpr int image_size = 224;
    constexpr int class_count = 1000;
    constexpr size_t input_elements = 3 * image_size * image_size;
    const std::string engine_path = "models/mobilenet_v3_small_fp16.engine";
    const std::string image_path = "assets/dog.jpg";

    try {
        cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
        if (image.empty()) {
            throw std::runtime_error("无法读取测试图片：" + image_path);
        }

        std::vector<std::string> labels;
        std::ifstream labels_file("assets/imagenet_classes.txt");
        std::string label;
        while (std::getline(labels_file, label)) {
            if (!label.empty() && label.back() == '\r') {
                label.pop_back();
            }
            labels.push_back(label);
        }
        if (labels.size() != class_count) {
            throw std::runtime_error("ImageNet 类别文件应包含 1000 行");
        }

        const double scale = 256.0 / std::min(image.rows, image.cols);
        const int resized_width = static_cast<int>(image.cols * scale);
        const int resized_height = static_cast<int>(image.rows * scale);
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(resized_width, resized_height), 0.0, 0.0, cv::INTER_LINEAR);
        const cv::Rect crop_rect(
            (resized.cols - image_size) / 2,
            (resized.rows - image_size) / 2,
            image_size,
            image_size);
        const cv::Mat crop = resized(crop_rect);

        constexpr float mean[3] = {0.485F, 0.456F, 0.406F};
        constexpr float standard_deviation[3] = {0.229F, 0.224F, 0.225F};
        std::vector<float> input(input_elements);
        for (int y = 0; y < image_size; ++y) {
            const cv::Vec3b* row = crop.ptr<cv::Vec3b>(y);
            for (int x = 0; x < image_size; ++x) {
                for (int channel = 0; channel < 3; ++channel) {
                    const float pixel = static_cast<float>(row[x][2 - channel]) / 255.0F;
                    input[(channel * image_size + y) * image_size + x] =
                        (pixel - mean[channel]) / standard_deviation[channel];
                }
            }
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
        std::vector<char> engine_data(static_cast<size_t>(engine_size));
        engine_file.seekg(0, std::ios::beg);
        if (!engine_file.read(engine_data.data(), engine_size)) {
            throw std::runtime_error("读取 TensorRT engine 失败");
        }
        std::unique_ptr<nvinfer1::ICudaEngine> engine{
            runtime->deserializeCudaEngine(engine_data.data(), engine_data.size())};
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
            throw std::runtime_error("预期 engine 恰有一个输入和一个输出 tensor");
        }
        const nvinfer1::Dims input_dims = engine->getTensorShape(input_name);
        const nvinfer1::Dims output_dims = engine->getTensorShape(output_name);
        if (input_dims.nbDims != 4 || input_dims.d[0] != 1 || input_dims.d[1] != 3 ||
            input_dims.d[2] != image_size || input_dims.d[3] != image_size ||
            output_dims.nbDims != 2 || output_dims.d[0] != 1 || output_dims.d[1] != class_count ||
            engine->getTensorDataType(input_name) != nvinfer1::DataType::kFLOAT ||
            engine->getTensorDataType(output_name) != nvinfer1::DataType::kFLOAT) {
            throw std::runtime_error("engine I/O 应为 FP32 [1,3,224,224] -> [1,1000]");
        }

        float* device_input = nullptr;
        float* device_output = nullptr;
        CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&device_input), input_elements * sizeof(float)));
        CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&device_output), class_count * sizeof(float)));
        if (!context->setTensorAddress(input_name, device_input) ||
            !context->setTensorAddress(output_name, device_output)) {
            throw std::runtime_error("TensorRT 输入 / 输出地址绑定失败");
        }
        CHECK_CUDA(cudaMemcpy(device_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
        if (!context->enqueueV3(nullptr)) {
            throw std::runtime_error("TensorRT 推理失败");
        }
        std::vector<float> logits(class_count);
        CHECK_CUDA(cudaMemcpy(logits.data(), device_output, logits.size() * sizeof(float), cudaMemcpyDeviceToHost));

        std::vector<int> indices(class_count);
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(indices.begin(), indices.begin() + 5, indices.end(),
                          [&logits](int left, int right) { return logits[left] > logits[right]; });
        const float max_logit = *std::max_element(logits.begin(), logits.end());
        double exp_sum = 0.0;
        for (float value : logits) {
            exp_sum += std::exp(static_cast<double>(value - max_logit));
        }

        std::cout << "Image: " << image_path << "\nTop-5:\n";
        for (int index = 0; index < 5; ++index) {
            const int class_id = indices[index];
            const double probability = std::exp(static_cast<double>(logits[class_id] - max_logit)) / exp_sum;
            std::cout << labels[class_id] << ": " << probability * 100.0 << "%\n";
        }

        CHECK_CUDA(cudaFree(device_input));
        CHECK_CUDA(cudaFree(device_output));
    } catch (const std::exception& error) {
        std::cerr << "错误：" << error.what() << '\n';
        return 1;
    }
    return 0;
}

#undef CHECK_CUDA
