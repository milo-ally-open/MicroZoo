#include "logger.h"
#include "postprocess.h"
#include "preprocess.h"

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
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

int main() {
    constexpr int input_size = 640;
    constexpr int prediction_count = 300;
    constexpr int prediction_elements = 6;
    constexpr int max_detections = 300;
    constexpr float confidence_threshold = 0.25F;
    constexpr float nms_threshold = 0.70F;
    constexpr size_t input_elements = 3 * input_size * input_size;
    const std::string engine_path = "models/yolo26n_fp16.engine";
    const std::string labels_path = "assets/coco_classes.txt";

    try {
        // 读取 COCO 类别表，类别索引与官方 YOLO26n checkpoint 一致。
        std::vector<std::string> labels;
        std::ifstream labels_file(labels_path);
        if (!labels_file) {
            throw std::runtime_error("无法打开 COCO 类别文件：" + labels_path);
        }
        std::string label;
        while (std::getline(labels_file, label)) {
            if (!label.empty() && label.back() == '\r') {
                label.pop_back();
            }
            labels.push_back(label);
        }
        if (labels.size() != 80) {
            throw std::runtime_error("COCO 类别文件应包含 80 行");
        }

        // 默认读取 USB/V4L2 摄像头 /dev/video0；如需其他设备，修改这里的索引。
        cv::VideoCapture camera(0, cv::CAP_V4L2);
        if (!camera.isOpened()) {
            camera.open(0);
        }
        if (!camera.isOpened()) {
            throw std::runtime_error("无法打开摄像头 /dev/video0");
        }
        camera.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        camera.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
        camera.set(cv::CAP_PROP_BUFFERSIZE, 1);
        cv::Mat frame;
        if (!camera.read(frame) || frame.empty()) {
            throw std::runtime_error("摄像头已打开，但无法读取第一帧");
        }
        if (!frame.isContinuous()) {
            frame = frame.clone();
        }
        const int camera_width = frame.cols;
        const int camera_height = frame.rows;
        const size_t frame_bytes = frame.total() * frame.elemSize();

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
            throw std::runtime_error("预期 engine 恰有一个输入 tensor 和一个输出 tensor");
        }
        const nvinfer1::Dims input_dims = engine->getTensorShape(input_name);
        const nvinfer1::Dims output_dims = engine->getTensorShape(output_name);
        if (input_dims.nbDims != 4 || input_dims.d[0] != 1 || input_dims.d[1] != 3 ||
            input_dims.d[2] != input_size || input_dims.d[3] != input_size ||
            output_dims.nbDims != 3 || output_dims.d[0] != 1 ||
            output_dims.d[1] != prediction_count || output_dims.d[2] != prediction_elements ||
            engine->getTensorDataType(input_name) != nvinfer1::DataType::kFLOAT ||
            engine->getTensorDataType(output_name) != nvinfer1::DataType::kFLOAT) {
            throw std::runtime_error("engine I/O 应为 FP32 [1,3,640,640] -> [1,300,6]");
        }

        cudaStream_t stream = nullptr;
        unsigned char* device_frame = nullptr;
        float* device_input = nullptr;
        float* device_predictions = nullptr;
        float* device_boxes = nullptr;
        int* device_box_count = nullptr;
        if (!checkCuda(cudaStreamCreate(&stream)) ||
            !checkCuda(cudaMalloc(reinterpret_cast<void**>(&device_frame), frame_bytes)) ||
            !checkCuda(cudaMalloc(reinterpret_cast<void**>(&device_input), input_elements * sizeof(float))) ||
            !checkCuda(cudaMalloc(reinterpret_cast<void**>(&device_predictions), prediction_count * prediction_elements * sizeof(float))) ||
            !checkCuda(cudaMalloc(reinterpret_cast<void**>(&device_boxes), max_detections * 7 * sizeof(float))) ||
            !checkCuda(cudaMalloc(reinterpret_cast<void**>(&device_box_count), sizeof(int)))) {
            throw std::runtime_error("CUDA buffer 初始化失败");
        }
        if (!context->setTensorAddress(input_name, device_input) ||
            !context->setTensorAddress(output_name, device_predictions)) {
            throw std::runtime_error("TensorRT 输入 / 输出地址绑定失败");
        }

        // 预热固定输入 shape 和 TensorRT context，避免把首次初始化时间算进实时帧延迟。
        if (!checkCuda(cudaMemsetAsync(device_input, 0, input_elements * sizeof(float), stream))) {
            throw std::runtime_error("TensorRT warmup 输入初始化失败");
        }
        for (int iteration = 0; iteration < 10; ++iteration) {
            if (!context->enqueueV3(stream)) {
                throw std::runtime_error("TensorRT warmup inference 失败");
            }
        }
        if (!checkCuda(cudaStreamSynchronize(stream))) {
            throw std::runtime_error("TensorRT warmup stream 同步失败");
        }

        std::cout << "YOLO26n TensorRT 实时检测已启动\n"
                  << "Camera: /dev/video0\n"
                  << "Camera resolution: " << camera_width << "x" << camera_height << "\n"
                  << "Input: 640x640, batch=1, FP16 engine\n"
                  << "CUDA preprocessing + TensorRT + CUDA NMS\n"
                  << "按 q 或 Esc 退出\n";

        int frame_index = 0;
        std::vector<float> host_boxes(max_detections * 7);
        while (!frame.empty()) {
            if (!frame.isContinuous()) {
                frame = frame.clone();
            }
            if (frame.cols != camera_width || frame.rows != camera_height) {
                throw std::runtime_error("摄像头分辨率在运行中发生变化");
            }

            const float scale = std::min(
                static_cast<float>(input_size) / frame.cols,
                static_cast<float>(input_size) / frame.rows);
            const int resized_width = static_cast<int>(std::round(frame.cols * scale));
            const int resized_height = static_cast<int>(std::round(frame.rows * scale));
            const int pad_left = (input_size - resized_width) / 2;
            const int pad_top = (input_size - resized_height) / 2;
            const auto end_to_end_start = std::chrono::steady_clock::now();

            // GPU kernel 将 BGR 摄像头帧直接变成 RGB NCHW 输入，包含 letterbox 与 /255。
            const auto preprocess_start = std::chrono::steady_clock::now();
            if (!checkCuda(cudaMemcpyAsync(device_frame, frame.data, frame_bytes, cudaMemcpyHostToDevice, stream)) ||
                !checkCuda(launch_preprocess_kernel(
                    device_frame, device_input, frame.cols, frame.rows, resized_width, resized_height,
                    pad_left, pad_top, stream)) ||
                !checkCuda(cudaStreamSynchronize(stream))) {
                throw std::runtime_error("摄像头帧 CUDA 预处理失败");
            }
            const double preprocess_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - preprocess_start).count();

            const auto inference_start = std::chrono::steady_clock::now();
            if (!context->enqueueV3(stream) || !checkCuda(cudaStreamSynchronize(stream))) {
                throw std::runtime_error("TensorRT 实时推理失败");
            }
            const double inference_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - inference_start).count();

            // 导出图输出 TopK 候选；此 CUDA 阶段做置信度过滤、逆 letterbox 和 class-aware NMS。
            const auto postprocess_start = std::chrono::steady_clock::now();
            if (!checkCuda(cudaMemsetAsync(device_box_count, 0, sizeof(int), stream)) ||
                !checkCuda(launch_decode_filter_kernel(
                    device_predictions, prediction_count, confidence_threshold, scale, pad_left, pad_top,
                    frame.cols, frame.rows, device_boxes, device_box_count, max_detections, stream)) ||
                !checkCuda(launch_fast_nms_kernel(
                    device_boxes, device_box_count, max_detections, nms_threshold, stream))) {
                throw std::runtime_error("CUDA 检测后处理 kernel 启动失败");
            }
            int host_box_count = 0;
            if (!checkCuda(cudaMemcpyAsync(&host_box_count, device_box_count, sizeof(int), cudaMemcpyDeviceToHost, stream)) ||
                !checkCuda(cudaMemcpyAsync(host_boxes.data(), device_boxes, host_boxes.size() * sizeof(float), cudaMemcpyDeviceToHost, stream)) ||
                !checkCuda(cudaStreamSynchronize(stream))) {
                throw std::runtime_error("读取 CUDA 检测结果失败");
            }
            const double postprocess_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - postprocess_start).count();
            const double end_to_end_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - end_to_end_start).count();

            const int valid_box_count = std::min(host_box_count, max_detections);
            int visible_detections = 0;
            for (int index = 0; index < valid_box_count; ++index) {
                const float* box = host_boxes.data() + index * 7;
                const int class_id = static_cast<int>(std::round(box[5]));
                if (box[6] < 0.5F || class_id < 0 || class_id >= static_cast<int>(labels.size())) {
                    continue;
                }
                ++visible_detections;
                const int left = std::clamp(static_cast<int>(std::round(box[0])), 0, frame.cols - 1);
                const int top = std::clamp(static_cast<int>(std::round(box[1])), 0, frame.rows - 1);
                const int right = std::clamp(static_cast<int>(std::round(box[2])), 0, frame.cols - 1);
                const int bottom = std::clamp(static_cast<int>(std::round(box[3])), 0, frame.rows - 1);
                cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 220, 0), 2);
                std::ostringstream caption;
                caption << labels[class_id] << " " << std::fixed << std::setprecision(2) << box[4];
                const int text_y = std::max(top - 6, 18);
                cv::putText(frame, caption.str(), cv::Point(left, text_y), cv::FONT_HERSHEY_SIMPLEX,
                            0.55, cv::Scalar(0, 220, 0), 2, cv::LINE_AA);
            }

            const double process_fps = 1000.0 / std::max(end_to_end_ms, 1.0e-6);
            std::ostringstream status;
            status << std::fixed << std::setprecision(1) << "PROC FPS " << process_fps
                   << " | E2E " << end_to_end_ms << " ms"
                   << " | Pre " << preprocess_ms << " ms"
                   << " | TRT " << inference_ms << " ms"
                   << " | Post " << postprocess_ms << " ms";
            cv::putText(frame, status.str(), cv::Point(16, 28), cv::FONT_HERSHEY_SIMPLEX,
                        0.62, cv::Scalar(30, 230, 255), 2, cv::LINE_AA);
            cv::imshow("YOLO26n TensorRT - press q to quit", frame);

            ++frame_index;
            if (frame_index % 30 == 0) {
                std::cout << "frame=" << frame_index
                          << " detections=" << visible_detections
                          << " preprocess=" << std::fixed << std::setprecision(2) << preprocess_ms
                          << " ms inference=" << inference_ms << " ms postprocess=" << postprocess_ms
                          << " ms end-to-end=" << end_to_end_ms << " ms process_fps=" << process_fps << std::endl;
            }

            const int key = cv::waitKey(1) & 0xFF;
            if (key == 'q' || key == 27) {
                break;
            }
            if (!camera.read(frame)) {
                break;
            }
        }

        cv::destroyAllWindows();
        camera.release();
        if (!checkCuda(cudaStreamSynchronize(stream)) ||
            !checkCuda(cudaFree(device_frame)) ||
            !checkCuda(cudaFree(device_input)) ||
            !checkCuda(cudaFree(device_predictions)) ||
            !checkCuda(cudaFree(device_boxes)) ||
            !checkCuda(cudaFree(device_box_count)) ||
            !checkCuda(cudaStreamDestroy(stream))) {
            throw std::runtime_error("CUDA cleanup 失败");
        }
    } catch (const std::exception& error) {
        std::cerr << "错误：" << error.what() << std::endl;
        return 1;
    }
    return 0;
}
