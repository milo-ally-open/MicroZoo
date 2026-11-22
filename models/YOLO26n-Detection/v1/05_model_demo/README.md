# YOLO26n TensorRT C++ v1 Demo

该 Demo 使用固定 `assets/sample.jpg` 做一次目标检测。OpenCV 在 CPU 上完成 RGB letterbox 和 NCHW 转换；TensorRT 执行 FP16 engine；C++ 在 CPU 上过滤候选框并做 class-aware NMS，再打印检测类别、置信度和原图坐标。没有摄像头、GUI、自定义 CUDA kernel、重复 benchmark 或 CUDA Graph。

## 构建与运行

需要 JetPack 提供的 TensorRT、CUDA Toolkit/runtime、OpenCV 4、CMake 和 C++17 编译器。在本目录执行：

```bash
cmake -S . -B build
cmake --build build -j2
./build/yolo26n_demo
```

程序从当前目录读取 `models/yolo26n_fp16.engine`、`assets/sample.jpg` 和 `assets/coco_classes.txt`。engine 由 `../02_model_convert/05-build_tensorrt.py` 根据完整 ONNX 构建；换设备或 TensorRT 版本时应重新构建并验证。
