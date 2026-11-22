# YOLO26n TensorRT C++ 检测 Demo

在 Jetson 上使用 TensorRT C++ API 加载本机生成的 YOLO26n FP16 engine，从 `/dev/video0` 实时读取摄像头画面并显示检测框、类别和帧率。

## 准备资源

在 ONNX 与 engine 构建成功后运行：

```bash
python3 00-prepare_assets.py
```

脚本从官方权重导出 80 类名称，并把 engine 复制到 `models/yolo26n_fp16.engine`。运行演示不依赖测试图片。

## 构建与运行

需要 JetPack 提供的 CUDA、TensorRT、OpenCV、CMake 和 C++17 编译器。在此目录执行：

```bash
cmake -S . -B build
cmake --build build -j2
./build/yolo26n_demo
```

`main.cpp` 使用 OpenCV 从摄像头采集 BGR 帧；`kernel/preprocess.cu` 在 GPU 完成 letterbox、RGB/NCHW 转换和 `[0,1]` 归一化；TensorRT 对每帧执行推理；CUDA 后处理核过滤低置信度框、把坐标映射回原图并按类别执行 IoU NMS。模型导出的 `[1,300,6]` 结果包含 TopK 解码但没有 NMS，NMS 阈值与置信度阈值分别固定为 0.70 和 0.25。窗口中按 `q` 或 Esc 退出。

实时窗口叠加检测类别、置信度、原图坐标、预处理时间、TensorRT 推理时间、CUDA 后处理时间、端到端延迟和处理 FPS；每 30 帧向终端打印一次汇总。处理 FPS 按端到端处理时长计算，不代表摄像头实际采集帧率。端到端计时从摄像头帧读取完成开始，包含预处理、TensorRT、CUDA NMS 和结果回传，不包含摄像头采集、框绘制与窗口刷新。

该 engine 在 Jetson Orin Nano Super、JetPack R36.4.3、TensorRT 10.3.0 上构建。TensorRT engine 与 ONNX 不跨设备或 TensorRT 版本保证可移植；需要在目标设备重新构建并验证。

本次摄像头实时运行记录见 [demo_report.md](demo_report.md)。
