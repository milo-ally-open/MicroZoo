#include "logger.h"

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct TrtDeleter {
    template <typename T>
    void operator()(T* object) const {
        delete object;
    }
};

struct CudaBuffers {
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    float* host_input = nullptr;
    float* host_output = nullptr;
    float* device_input = nullptr;
    float* device_output = nullptr;

    ~CudaBuffers() {
        if (stream != nullptr) {
            cudaStreamSynchronize(stream);
        }
        if (graph_exec != nullptr) {
            cudaGraphExecDestroy(graph_exec);
        }
        if (graph != nullptr) {
            cudaGraphDestroy(graph);
        }
        if (device_input != nullptr) {
            cudaFree(device_input);
        }
        if (device_output != nullptr) {
            cudaFree(device_output);
        }
        if (host_input != nullptr) {
            cudaFreeHost(host_input);
        }
        if (host_output != nullptr) {
            cudaFreeHost(host_output);
        }
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
        }
    }
};

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

std::vector<std::string> readLabels(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("无法打开类别文件：" + path);
    }

    std::vector<std::string> labels;
    std::string label;
    while (std::getline(file, label)) {
        if (!label.empty() && label.back() == '\r') {
            label.pop_back();
        }
        labels.push_back(label);
    }
    if (labels.size() != 1000) {
        throw std::runtime_error("ImageNet 类别文件应包含 1000 行，实际为 " + std::to_string(labels.size()));
    }
    return labels;
}

void preprocessImage(const cv::Mat& image, float* input) {
    if (image.empty()) {
        throw std::runtime_error("无法读取输入图像");
    }

    constexpr int resize_size = 256;
    constexpr int crop_size = 224;
    const double scale = static_cast<double>(resize_size) / std::min(image.rows, image.cols);
    const int resized_width = static_cast<int>(image.cols * scale);
    const int resized_height = static_cast<int>(image.rows * scale);
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(resized_width, resized_height), 0.0, 0.0, cv::INTER_LINEAR);

    const int left = (resized.cols - crop_size) / 2;
    const int top = (resized.rows - crop_size) / 2;
    const cv::Mat crop = resized(cv::Rect(left, top, crop_size, crop_size));
    constexpr float mean[3] = {0.485F, 0.456F, 0.406F};
    constexpr float standard_deviation[3] = {0.229F, 0.224F, 0.225F};

    for (int y = 0; y < crop_size; ++y) {
        const cv::Vec3b* row = crop.ptr<cv::Vec3b>(y);
        for (int x = 0; x < crop_size; ++x) {
            for (int channel = 0; channel < 3; ++channel) {
                const float rgb_value = static_cast<float>(row[x][2 - channel]) / 255.0F;
                const int index = (channel * crop_size + y) * crop_size + x;
                input[index] = (rgb_value - mean[channel]) / standard_deviation[channel];
            }
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc > 4) {
            std::cerr << "用法：" << argv[0] << " [image.jpg] [engine] [imagenet_classes.txt]\n";
            return 2;
        }

        const std::string image_path = argc > 1 ? argv[1] : "assets/dog.jpg";
        const std::string engine_path = argc > 2 ? argv[2] : "models/mobilenet_v3_small_fp16.engine";
        const std::string labels_path = argc > 3 ? argv[3] : "assets/imagenet_classes.txt";
        const auto setup_started = std::chrono::steady_clock::now();

        demo::Logger logger;
        std::unique_ptr<nvinfer1::IRuntime, TrtDeleter> runtime{nvinfer1::createInferRuntime(logger)};
        if (!runtime) {
            throw std::runtime_error("创建 TensorRT runtime 失败");
        }

        std::ifstream engine_file(engine_path, std::ios::binary);
        if (!engine_file) {
            throw std::runtime_error("无法打开 TensorRT engine：" + engine_path);
        }
        std::vector<char> engine_data(
            (std::istreambuf_iterator<char>(engine_file)),
            std::istreambuf_iterator<char>()
        );
        std::unique_ptr<nvinfer1::ICudaEngine, TrtDeleter> engine{
            runtime->deserializeCudaEngine(engine_data.data(), engine_data.size())
        };
        if (!engine) {
            throw std::runtime_error("TensorRT 无法反序列化 engine：" + engine_path);
        }

        std::unique_ptr<nvinfer1::IExecutionContext, TrtDeleter> context{engine->createExecutionContext()};
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
            input_dims.d[2] != 224 || input_dims.d[3] != 224 ||
            output_dims.nbDims != 2 || output_dims.d[0] != 1 || output_dims.d[1] != 1000) {
            throw std::runtime_error("Engine shape 与 demo 预期不符；要求输入 [1,3,224,224]、输出 [1,1000]");
        }
        if (engine->getTensorDataType(input_name) != nvinfer1::DataType::kFLOAT ||
            engine->getTensorDataType(output_name) != nvinfer1::DataType::kFLOAT) {
            throw std::runtime_error("当前 demo 要求 engine 的输入和输出为 FP32 I/O（engine 内部可使用 FP16）");
        }

        constexpr size_t input_elements = 3 * 224 * 224;
        constexpr size_t output_elements = 1000;
        constexpr size_t input_bytes = input_elements * sizeof(float);
        constexpr size_t output_bytes = output_elements * sizeof(float);
        CudaBuffers buffers;
        checkCuda(cudaStreamCreate(&buffers.stream), "创建 CUDA stream");
        checkCuda(cudaHostAlloc(reinterpret_cast<void**>(&buffers.host_input), input_bytes, cudaHostAllocDefault), "分配 pinned host input");
        checkCuda(cudaHostAlloc(reinterpret_cast<void**>(&buffers.host_output), output_bytes, cudaHostAllocDefault), "分配 pinned host output");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&buffers.device_input), input_bytes), "分配 device input");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&buffers.device_output), output_bytes), "分配 device output");
        if (!context->setTensorAddress(input_name, buffers.device_input) ||
            !context->setTensorAddress(output_name, buffers.device_output)) {
            throw std::runtime_error("TensorRT 绑定输入或输出 buffer 失败");
        }

        checkCuda(cudaMemsetAsync(buffers.device_input, 0, input_bytes, buffers.stream), "初始化 device input");
        for (int iteration = 0; iteration < 10; ++iteration) {
            if (!context->enqueueV3(buffers.stream)) {
                throw std::runtime_error("TensorRT warmup inference 执行失败");
            }
        }
        checkCuda(cudaStreamSynchronize(buffers.stream), "同步 TensorRT warmup");

        checkCuda(cudaStreamBeginCapture(buffers.stream, cudaStreamCaptureModeGlobal), "开始 CUDA Graph capture");
        const bool captured = context->enqueueV3(buffers.stream);
        const cudaError_t capture_status = cudaStreamEndCapture(buffers.stream, &buffers.graph);
        checkCuda(capture_status, "结束 CUDA Graph capture");
        if (!captured || buffers.graph == nullptr) {
            throw std::runtime_error("捕获 TensorRT 推理到 CUDA Graph 失败");
        }
        checkCuda(cudaGraphInstantiate(&buffers.graph_exec, buffers.graph, 0), "实例化 CUDA Graph");
        for (int iteration = 0; iteration < 5; ++iteration) {
            checkCuda(cudaGraphLaunch(buffers.graph_exec, buffers.stream), "CUDA Graph warmup replay");
        }
        checkCuda(cudaStreamSynchronize(buffers.stream), "同步 CUDA Graph warmup");

        const double setup_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - setup_started
        ).count();
        const std::vector<std::string> labels = readLabels(labels_path);

        const auto request_started = std::chrono::steady_clock::now();
        const cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
        const auto image_loaded = std::chrono::steady_clock::now();
        if (image.empty()) {
            throw std::runtime_error("无法读取输入图像：" + image_path);
        }
        preprocessImage(image, buffers.host_input);
        const auto image_preprocessed = std::chrono::steady_clock::now();

        checkCuda(cudaMemcpyAsync(
            buffers.device_input, buffers.host_input, input_bytes, cudaMemcpyHostToDevice, buffers.stream
        ), "复制输入到 GPU");
        checkCuda(cudaGraphLaunch(buffers.graph_exec, buffers.stream), "运行 CUDA Graph inference");
        checkCuda(cudaMemcpyAsync(
            buffers.host_output, buffers.device_output, output_bytes, cudaMemcpyDeviceToHost, buffers.stream
        ), "复制输出到 CPU");
        checkCuda(cudaStreamSynchronize(buffers.stream), "等待推理完成");
        const auto inference_finished = std::chrono::steady_clock::now();

        const float max_logit = *std::max_element(buffers.host_output, buffers.host_output + output_elements);
        double exp_sum = 0.0;
        for (size_t index = 0; index < output_elements; ++index) {
            exp_sum += std::exp(static_cast<double>(buffers.host_output[index] - max_logit));
        }
        std::vector<int> indices(output_elements);
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(indices.begin(), indices.begin() + 5, indices.end(), [&](int left, int right) {
            return buffers.host_output[left] > buffers.host_output[right];
        });
        const auto postprocess_finished = std::chrono::steady_clock::now();

        const auto milliseconds = [](auto begin, auto end) {
            return std::chrono::duration<double, std::milli>(end - begin).count();
        };
        std::cout << "Engine: " << engine_path << '\n';
        std::cout << "Image: " << image_path << " (" << image.cols << 'x' << image.rows << ")\n";
        std::cout << "Execution: TensorRT FP16 engine + CUDA Graph replay\n";
        std::cout << "Top-5:\n";
        for (int rank = 0; rank < 5; ++rank) {
            const int index = indices[rank];
            const double probability = std::exp(static_cast<double>(buffers.host_output[index] - max_logit)) / exp_sum;
            std::cout << "  " << rank + 1 << ". " << labels[index]
                      << " (class " << index << ", " << probability * 100.0 << "%)\n";
        }
        std::cout << "Timing (ms): setup=" << setup_ms
                  << ", image_load=" << milliseconds(request_started, image_loaded)
                  << ", preprocess=" << milliseconds(image_loaded, image_preprocessed)
                  << ", H2D+Graph+D2H=" << milliseconds(image_preprocessed, inference_finished)
                  << ", postprocess=" << milliseconds(inference_finished, postprocess_finished)
                  << ", request_total=" << milliseconds(request_started, postprocess_finished) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
