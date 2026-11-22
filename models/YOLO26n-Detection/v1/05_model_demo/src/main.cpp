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

struct Detection {
    float x1;
    float y1;
    float x2;
    float y2;
    float confidence;
    int class_id;
};

int main() {
    constexpr int input_size = 640;
    constexpr int candidate_count = 300;
    constexpr int candidate_size = 6;
    constexpr int class_count = 80;
    constexpr float confidence_threshold = 0.25F;
    constexpr float iou_threshold = 0.70F;
    constexpr size_t input_elements = 3 * input_size * input_size;
    const std::string engine_path = "models/yolo26n_fp16.engine";
    const std::string image_path = "assets/sample.jpg";

    try {
        cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
        if (image.empty()) {
            throw std::runtime_error("无法读取测试图片：" + image_path);
        }

        std::vector<std::string> labels;
        std::ifstream labels_file("assets/coco_classes.txt");
        std::string label;
        while (std::getline(labels_file, label)) {
            if (!label.empty() && label.back() == '\r') {
                label.pop_back();
            }
            labels.push_back(label);
        }
        if (labels.size() != class_count) {
            throw std::runtime_error("COCO 类别文件应包含 80 行");
        }

        const float scale = std::min(
            static_cast<float>(input_size) / image.cols,
            static_cast<float>(input_size) / image.rows);
        const int resized_width = static_cast<int>(std::round(image.cols * scale));
        const int resized_height = static_cast<int>(std::round(image.rows * scale));
        const int pad_left = (input_size - resized_width) / 2;
        const int pad_top = (input_size - resized_height) / 2;
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(resized_width, resized_height), 0.0, 0.0, cv::INTER_LINEAR);
        cv::Mat letterboxed(input_size, input_size, CV_8UC3, cv::Scalar(114, 114, 114));
        resized.copyTo(letterboxed(cv::Rect(pad_left, pad_top, resized_width, resized_height)));
        cv::Mat rgb;
        cv::cvtColor(letterboxed, rgb, cv::COLOR_BGR2RGB);
        rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);

        std::vector<float> input(input_elements);
        for (int y = 0; y < input_size; ++y) {
            const cv::Vec3f* row = rgb.ptr<cv::Vec3f>(y);
            for (int x = 0; x < input_size; ++x) {
                for (int channel = 0; channel < 3; ++channel) {
                    input[(channel * input_size + y) * input_size + x] = row[x][channel];
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
            input_dims.d[2] != input_size || input_dims.d[3] != input_size ||
            output_dims.nbDims != 3 || output_dims.d[0] != 1 ||
            output_dims.d[1] != candidate_count || output_dims.d[2] != candidate_size ||
            engine->getTensorDataType(input_name) != nvinfer1::DataType::kFLOAT ||
            engine->getTensorDataType(output_name) != nvinfer1::DataType::kFLOAT) {
            throw std::runtime_error("engine I/O 应为 FP32 [1,3,640,640] -> [1,300,6]");
        }

        float* device_input = nullptr;
        float* device_output = nullptr;
        CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&device_input), input_elements * sizeof(float)));
        CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&device_output), candidate_count * candidate_size * sizeof(float)));
        if (!context->setTensorAddress(input_name, device_input) ||
            !context->setTensorAddress(output_name, device_output)) {
            throw std::runtime_error("TensorRT 输入 / 输出地址绑定失败");
        }
        CHECK_CUDA(cudaMemcpy(device_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
        if (!context->enqueueV3(nullptr)) {
            throw std::runtime_error("TensorRT 推理失败");
        }
        std::vector<float> predictions(candidate_count * candidate_size);
        CHECK_CUDA(cudaMemcpy(predictions.data(), device_output, predictions.size() * sizeof(float), cudaMemcpyDeviceToHost));

        std::vector<Detection> candidates;
        for (int index = 0; index < candidate_count; ++index) {
            const float* row = predictions.data() + index * candidate_size;
            const int class_id = static_cast<int>(std::round(row[5]));
            if (row[4] < confidence_threshold || class_id < 0 || class_id >= class_count) {
                continue;
            }
            candidates.push_back({
                std::clamp((row[0] - pad_left) / scale, 0.0F, static_cast<float>(image.cols)),
                std::clamp((row[1] - pad_top) / scale, 0.0F, static_cast<float>(image.rows)),
                std::clamp((row[2] - pad_left) / scale, 0.0F, static_cast<float>(image.cols)),
                std::clamp((row[3] - pad_top) / scale, 0.0F, static_cast<float>(image.rows)),
                row[4],
                class_id,
            });
        }
        std::sort(candidates.begin(), candidates.end(), [](const Detection& left, const Detection& right) {
            return left.confidence > right.confidence;
        });

        std::vector<Detection> detections;
        for (const Detection& candidate : candidates) {
            bool suppressed = false;
            for (const Detection& kept : detections) {
                if (candidate.class_id != kept.class_id) {
                    continue;
                }
                const float intersection_width = std::max(0.0F, std::min(candidate.x2, kept.x2) - std::max(candidate.x1, kept.x1));
                const float intersection_height = std::max(0.0F, std::min(candidate.y2, kept.y2) - std::max(candidate.y1, kept.y1));
                const float intersection = intersection_width * intersection_height;
                const float candidate_area = std::max(0.0F, candidate.x2 - candidate.x1) * std::max(0.0F, candidate.y2 - candidate.y1);
                const float kept_area = std::max(0.0F, kept.x2 - kept.x1) * std::max(0.0F, kept.y2 - kept.y1);
                const float iou = intersection / std::max(candidate_area + kept_area - intersection, 1.0e-12F);
                if (iou > iou_threshold) {
                    suppressed = true;
                    break;
                }
            }
            if (!suppressed) {
                detections.push_back(candidate);
            }
        }

        std::cout << "Image: " << image_path << "\nDetections: " << detections.size() << '\n';
        for (const Detection& detection : detections) {
            std::cout << labels[detection.class_id] << " " << detection.confidence
                      << " [" << detection.x1 << ", " << detection.y1 << ", "
                      << detection.x2 << ", " << detection.y2 << "]\n";
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
