#include "logger.h"
#include "preprocess.h"
#include "postprocess.h"

#include <NvInfer.h>
#include <cuda_runtime_api.h>
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

bool __check_cuda_runtime(cudaError_t code, const char* operation, const char* file, int line) {
    if (code == cudaSuccess) {
        return true;
    }

    const char* error_name = cudaGetErrorName(code);
    const char* error_message = cudaGetErrorString(code);
    std::fprintf(stderr, "%s:%d %s failed.\n  code = %s, message = %s\n",
                 file, line, operation,
                 error_name == nullptr ? "unknown" : error_name,
                 error_message == nullptr ? "unknown" : error_message);
    return false;
}

#define CHECK_CUDA(operation) __check_cuda_runtime((operation), #operation, __FILE__, __LINE__)

int main() {
    constexpr int input_size = 640;
    constexpr int candidate_count = 300;
    constexpr int candidate_size = 6;
    constexpr int class_count = 80;
    constexpr float confidence_threshold = 0.25F;
    constexpr float iou_threshold = 0.70F;
    constexpr size_t input_elements = 3 * input_size * input_size;
    const std::string engine_path = "models/yolo26n_pruned_int8.engine";
    // 沿用 v1 已确认的 UVC 模式；固定 caps 避免相机自动协商到低帧率格式。
    // nvvidconv compute-hw=2 选择 Jetson VIC，appsink 只保留最新一帧。
    const std::string camera_pipeline =
        "v4l2src device=/dev/video0 io-mode=2 ! "
        "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
        "nvvidconv compute-hw=2 ! video/x-raw,format=BGRx ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "appsink max-buffers=1 drop=true sync=false";

    float* device_input = nullptr;
    float* device_output = nullptr;
    Detection* device_detections = nullptr;
    Detection* device_kept_detections = nullptr;
    int* device_valid = nullptr;
    int* device_detection_count = nullptr;
    unsigned char* device_suppression_mask = nullptr;
    unsigned char* device_frame = nullptr;
    unsigned int* device_box_mask = nullptr;
    size_t device_frame_capacity = 0;
    size_t device_box_mask_capacity = 0;
    cudaStream_t stream = nullptr;
    cudaEvent_t inference_start = nullptr;
    cudaEvent_t inference_stop = nullptr;
    cudaEvent_t preprocess_start = nullptr;
    cudaEvent_t preprocess_stop = nullptr;
    cudaEvent_t postprocess_start = nullptr;
    cudaEvent_t postprocess_stop = nullptr;

    try {
        std::vector<std::string> labels;
        std::ifstream labels_file("assets/coco_classes.txt");
        if (!labels_file) {
            throw std::runtime_error("无法打开 assets/coco_classes.txt");
        }
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
            throw std::runtime_error("TensorRT engine 文件为空：" + engine_path);
        }
        std::vector<char> engine_data(static_cast<size_t>(engine_size));
        engine_file.seekg(0, std::ios::beg);
        if (!engine_file.read(engine_data.data(), engine_size)) {
            throw std::runtime_error("读取 TensorRT engine 失败");
        }

        std::unique_ptr<nvinfer1::ICudaEngine> engine{
            runtime->deserializeCudaEngine(engine_data.data(), engine_data.size())};
        if (!engine) {
            throw std::runtime_error("反序列化 TensorRT engine 失败；请用当前 TensorRT 版本重新构建 engine");
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

        if (!CHECK_CUDA(cudaStreamCreate(&stream))) {
            throw std::runtime_error("创建 CUDA stream 失败");
        }
        if (!CHECK_CUDA(cudaEventCreate(&inference_start)) ||
            !CHECK_CUDA(cudaEventCreate(&inference_stop)) ||
            !CHECK_CUDA(cudaEventCreate(&preprocess_start)) ||
            !CHECK_CUDA(cudaEventCreate(&preprocess_stop)) ||
            !CHECK_CUDA(cudaEventCreate(&postprocess_start)) ||
            !CHECK_CUDA(cudaEventCreate(&postprocess_stop))) {
            throw std::runtime_error("创建 CUDA timing event 失败");
        }
        if (!CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&device_input), input_elements * sizeof(float))) ||
            !CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&device_output), candidate_count * candidate_size * sizeof(float))) ||
            !CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&device_detections), candidate_count * sizeof(Detection))) ||
            !CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&device_kept_detections), candidate_count * sizeof(Detection))) ||
            !CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&device_valid), candidate_count * sizeof(int))) ||
            !CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&device_detection_count), sizeof(int))) ||
            !CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&device_suppression_mask), candidate_count * candidate_count * sizeof(unsigned char)))) {
            throw std::runtime_error("分配 CUDA buffer 失败");
        }
        if (!context->setTensorAddress(input_name, device_input) ||
            !context->setTensorAddress(output_name, device_output)) {
            throw std::runtime_error("TensorRT 输入 / 输出地址绑定失败");
        }

        cv::VideoCapture camera(camera_pipeline, cv::CAP_GSTREAMER);
        if (!camera.isOpened()) {
            throw std::runtime_error("无法通过 GStreamer 打开 /dev/video0；请检查摄像头和 GStreamer 插件");
        }

        std::cout << "TensorRT INT8 engine: " << engine_path << '\n'
                  << "GStreamer pipeline: " << camera_pipeline << '\n'
                  << "Camera mode request: YUY2 640x480 @ 30 FPS via V4L2 + VIC | negotiated: "
                  << camera.get(cv::CAP_PROP_FRAME_WIDTH) << 'x'
                  << camera.get(cv::CAP_PROP_FRAME_HEIGHT) << " @ "
                  << camera.get(cv::CAP_PROP_FPS) << " FPS（按 q 或 Esc 退出）\n";
        cv::namedWindow("YOLO26n TensorRT INT8", cv::WINDOW_NORMAL);

        std::vector<Detection> host_detections(candidate_count);
        std::vector<Detection> detections;
        int host_detection_count = 0;
        bool camera_info_printed = false;
        auto fps_window_start = std::chrono::steady_clock::now();
        int frames_in_window = 0;
        double inference_ms_in_window = 0.0;
        double preprocess_ms_in_window = 0.0;
        double postprocess_ms_in_window = 0.0;
        double end_to_end_ms_in_window = 0.0;
        double camera_read_ms_in_window = 0.0;
        double display_ms_in_window = 0.0;
        double displayed_fps = 0.0;
        double displayed_inference_ms = 0.0;
        double displayed_preprocess_ms = 0.0;
        double displayed_postprocess_ms = 0.0;
        double displayed_end_to_end_ms = 0.0;
        double displayed_camera_read_ms = 0.0;
        double displayed_display_ms = 0.0;

        while (true) {
            const auto frame_start = std::chrono::steady_clock::now();
            cv::Mat frame;
            if (!camera.read(frame) || frame.empty()) {
                std::cerr << "摄像头读取失败，结束实时推理。\n";
                break;
            }
            const auto camera_read_end = std::chrono::steady_clock::now();
            const double camera_read_ms =
                std::chrono::duration<double, std::milli>(camera_read_end - frame_start).count();
            if (!camera_info_printed) {
                std::cout << "Camera frame: " << frame.cols << 'x' << frame.rows << '\n';
                camera_info_printed = true;
            }
            if (frame.type() != CV_8UC3) {
                throw std::runtime_error("摄像头应输出 8 位 BGR 三通道画面");
            }

            const size_t frame_bytes = static_cast<size_t>(frame.cols) * frame.rows * 3;
            const size_t box_mask_bytes = static_cast<size_t>(frame.cols) * frame.rows * sizeof(unsigned int);
            if (frame_bytes > device_frame_capacity) {
                if (device_frame != nullptr && !CHECK_CUDA(cudaFree(device_frame))) {
                    throw std::runtime_error("释放旧摄像头画面 CUDA buffer 失败");
                }
                if (!CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&device_frame), frame_bytes))) {
                    throw std::runtime_error("分配摄像头画面 CUDA buffer 失败");
                }
                device_frame_capacity = frame_bytes;
            }
            if (box_mask_bytes > device_box_mask_capacity) {
                if (device_box_mask != nullptr && !CHECK_CUDA(cudaFree(device_box_mask))) {
                    throw std::runtime_error("释放旧画框 mask CUDA buffer 失败");
                }
                if (!CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&device_box_mask), box_mask_bytes))) {
                    throw std::runtime_error("分配画框 mask CUDA buffer 失败");
                }
                device_box_mask_capacity = box_mask_bytes;
            }

            // 用等比例缩放和 114 灰边维持训练时的 640×640 letterbox 输入。
            const float scale = std::min(
                static_cast<float>(input_size) / frame.cols,
                static_cast<float>(input_size) / frame.rows);
            const int resized_width = static_cast<int>(std::round(frame.cols * scale));
            const int resized_height = static_cast<int>(std::round(frame.rows * scale));
            const int pad_left = (input_size - resized_width) / 2;
            const int pad_top = (input_size - resized_height) / 2;

            // 上传原始 BGR 帧；预处理和 CUDA 画框都复用这份 GPU 图像。
            if (!CHECK_CUDA(cudaMemcpy2DAsync(
                    device_frame, static_cast<size_t>(frame.cols) * 3,
                    frame.data, frame.step, static_cast<size_t>(frame.cols) * 3, frame.rows,
                    cudaMemcpyHostToDevice, stream))) {
                throw std::runtime_error("上传摄像头画面失败");
            }
            if (!CHECK_CUDA(cudaEventRecord(preprocess_start, stream)) ||
                !CHECK_CUDA(preprocess_letterbox_cuda(
                    device_frame, device_input, frame.cols, frame.rows,
                    static_cast<size_t>(frame.cols) * 3, input_size,
                    resized_width, resized_height, pad_left, pad_top, stream)) ||
                !CHECK_CUDA(cudaEventRecord(preprocess_stop, stream))) {
                throw std::runtime_error("CUDA 图像预处理失败");
            }
            if (!CHECK_CUDA(cudaEventRecord(inference_start, stream))) {
                throw std::runtime_error("记录 TensorRT 推理起始事件失败");
            }
            if (!context->enqueueV3(stream)) {
                throw std::runtime_error("TensorRT 推理失败");
            }
            if (!CHECK_CUDA(cudaEventRecord(inference_stop, stream))) {
                throw std::runtime_error("记录 TensorRT 推理结束事件失败");
            }

            // CUDA 完成置信度过滤、坐标还原、两两 IoU 判断和 greedy class-aware NMS。
            if (!CHECK_CUDA(cudaEventRecord(postprocess_start, stream))) {
                throw std::runtime_error("记录 CUDA 后处理起始事件失败");
            }
            if (!CHECK_CUDA(postprocess_cuda(
                    device_output, device_detections, device_valid, device_suppression_mask,
                    device_kept_detections, device_detection_count,
                    candidate_count, class_count, confidence_threshold, iou_threshold,
                    scale, pad_left, pad_top, frame.cols, frame.rows, stream))) {
                throw std::runtime_error("启动 CUDA 后处理 kernel 失败");
            }
            if (!CHECK_CUDA(draw_boxes_cuda(
                    device_frame, device_box_mask, device_kept_detections, device_detection_count,
                    candidate_count, frame.cols, frame.rows, static_cast<size_t>(frame.cols) * 3, stream))) {
                throw std::runtime_error("启动 CUDA 画框 kernel 失败");
            }
            if (!CHECK_CUDA(cudaEventRecord(postprocess_stop, stream))) {
                throw std::runtime_error("记录 CUDA 后处理结束事件失败");
            }
            if (!CHECK_CUDA(cudaMemcpyAsync(host_detections.data(), device_kept_detections,
                                            host_detections.size() * sizeof(Detection),
                                            cudaMemcpyDeviceToHost, stream)) ||
                !CHECK_CUDA(cudaMemcpyAsync(&host_detection_count, device_detection_count,
                                            sizeof(int), cudaMemcpyDeviceToHost, stream)) ||
                !CHECK_CUDA(cudaMemcpy2DAsync(
                    frame.data, frame.step, device_frame, static_cast<size_t>(frame.cols) * 3,
                    static_cast<size_t>(frame.cols) * 3, frame.rows,
                    cudaMemcpyDeviceToHost, stream)) ||
                !CHECK_CUDA(cudaStreamSynchronize(stream))) {
                throw std::runtime_error("读取 CUDA NMS / 画框结果失败");
            }
            float inference_ms = 0.0F;
            if (!CHECK_CUDA(cudaEventElapsedTime(&inference_ms, inference_start, inference_stop))) {
                throw std::runtime_error("读取 TensorRT 推理时间失败");
            }
            float preprocess_ms = 0.0F;
            if (!CHECK_CUDA(cudaEventElapsedTime(&preprocess_ms, preprocess_start, preprocess_stop))) {
                throw std::runtime_error("读取 CUDA 预处理时间失败");
            }
            float postprocess_ms = 0.0F;
            if (!CHECK_CUDA(cudaEventElapsedTime(&postprocess_ms, postprocess_start, postprocess_stop))) {
                throw std::runtime_error("读取 CUDA 后处理时间失败");
            }

            if (host_detection_count < 0 || host_detection_count > candidate_count) {
                throw std::runtime_error("CUDA NMS 返回了无效检测数量");
            }
            detections.clear();
            detections.assign(host_detections.begin(), host_detections.begin() + host_detection_count);

            const auto display_start = std::chrono::steady_clock::now();
            for (const Detection& detection : detections) {
                const cv::Point top_left(static_cast<int>(std::round(detection.x1)),
                                         static_cast<int>(std::round(detection.y1)));
                std::ostringstream caption;
                caption << labels[detection.class_id] << " " << std::fixed << std::setprecision(2)
                        << detection.confidence;
                cv::putText(frame, caption.str(), cv::Point(top_left.x, std::max(18, top_left.y - 5)),
                            cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 0), 2);
            }

            std::ostringstream fps_text;
            fps_text << "FPS: " << std::fixed << std::setprecision(1) << displayed_fps;
            cv::putText(frame, fps_text.str(), cv::Point(16, 30), cv::FONT_HERSHEY_SIMPLEX,
                        0.8, cv::Scalar(0, 255, 255), 2);
            std::ostringstream inference_text;
            inference_text << "TRT: " << std::fixed << std::setprecision(2) << displayed_inference_ms
                           << " | PRE: " << std::setprecision(2) << displayed_preprocess_ms
                           << " | POST: " << std::setprecision(2) << displayed_postprocess_ms << " ms";
            cv::putText(frame, inference_text.str(), cv::Point(16, 60), cv::FONT_HERSHEY_SIMPLEX,
                        0.65, cv::Scalar(0, 255, 255), 2);
            std::ostringstream end_to_end_text;
            end_to_end_text << "E2E: " << std::fixed << std::setprecision(1)
                            << displayed_end_to_end_ms << " ms | detections: " << detections.size();
            cv::putText(frame, end_to_end_text.str(), cv::Point(16, 90), cv::FONT_HERSHEY_SIMPLEX,
                        0.65, cv::Scalar(0, 255, 255), 2);
            std::ostringstream io_timing_text;
            io_timing_text << "READ: " << std::fixed << std::setprecision(1)
                           << displayed_camera_read_ms << " | SHOW: " << displayed_display_ms << " ms";
            cv::putText(frame, io_timing_text.str(), cv::Point(16, 120), cv::FONT_HERSHEY_SIMPLEX,
                        0.65, cv::Scalar(0, 255, 255), 2);
            cv::imshow("YOLO26n TensorRT INT8", frame);
            const int key = cv::waitKey(1);
            const auto frame_end = std::chrono::steady_clock::now();
            const double display_ms =
                std::chrono::duration<double, std::milli>(frame_end - display_start).count();
            const double end_to_end_ms =
                std::chrono::duration<double, std::milli>(frame_end - frame_start).count();

            ++frames_in_window;
            inference_ms_in_window += inference_ms;
            preprocess_ms_in_window += preprocess_ms;
            postprocess_ms_in_window += postprocess_ms;
            end_to_end_ms_in_window += end_to_end_ms;
            camera_read_ms_in_window += camera_read_ms;
            display_ms_in_window += display_ms;
            const auto now = std::chrono::steady_clock::now();
            const double window_seconds = std::chrono::duration<double>(now - fps_window_start).count();
            if (window_seconds >= 1.0) {
                displayed_fps = frames_in_window / window_seconds;
                displayed_inference_ms = inference_ms_in_window / frames_in_window;
                displayed_preprocess_ms = preprocess_ms_in_window / frames_in_window;
                displayed_postprocess_ms = postprocess_ms_in_window / frames_in_window;
                displayed_end_to_end_ms = end_to_end_ms_in_window / frames_in_window;
                displayed_camera_read_ms = camera_read_ms_in_window / frames_in_window;
                displayed_display_ms = display_ms_in_window / frames_in_window;
                std::cout << "FPS: " << std::fixed << std::setprecision(1) << displayed_fps
                          << " | TensorRT: " << std::setprecision(2) << displayed_inference_ms
                          << " ms | preprocess: " << std::setprecision(2) << displayed_preprocess_ms
                          << " ms | postprocess: " << std::setprecision(2) << displayed_postprocess_ms
                          << " ms | E2E: " << std::setprecision(1) << displayed_end_to_end_ms
                          << " ms | read: " << displayed_camera_read_ms
                          << " ms | show: " << displayed_display_ms
                          << " ms | detections: " << detections.size() << '\n';
                fps_window_start = now;
                frames_in_window = 0;
                inference_ms_in_window = 0.0;
                preprocess_ms_in_window = 0.0;
                postprocess_ms_in_window = 0.0;
                end_to_end_ms_in_window = 0.0;
                camera_read_ms_in_window = 0.0;
                display_ms_in_window = 0.0;
            }
            if (key == 'q' || key == 27) {
                break;
            }
        }

        cv::destroyAllWindows();
        CHECK_CUDA(cudaEventDestroy(inference_start));
        CHECK_CUDA(cudaEventDestroy(inference_stop));
        CHECK_CUDA(cudaEventDestroy(preprocess_start));
        CHECK_CUDA(cudaEventDestroy(preprocess_stop));
        CHECK_CUDA(cudaEventDestroy(postprocess_start));
        CHECK_CUDA(cudaEventDestroy(postprocess_stop));
        CHECK_CUDA(cudaFree(device_input));
        CHECK_CUDA(cudaFree(device_output));
        CHECK_CUDA(cudaFree(device_detections));
        CHECK_CUDA(cudaFree(device_kept_detections));
        CHECK_CUDA(cudaFree(device_valid));
        CHECK_CUDA(cudaFree(device_detection_count));
        CHECK_CUDA(cudaFree(device_suppression_mask));
        CHECK_CUDA(cudaFree(device_frame));
        CHECK_CUDA(cudaFree(device_box_mask));
        CHECK_CUDA(cudaStreamDestroy(stream));
    } catch (const std::exception& error) {
        std::cerr << "错误：" << error.what() << '\n';
        return 1;
    }
    return 0;
}
