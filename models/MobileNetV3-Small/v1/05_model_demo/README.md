# MobileNetV3-Small C++ v1 Demo

该 Demo 对固定的 `assets/dog.jpg` 做一次 MobileNetV3-Small 分类。OpenCV 在 CPU 上完成 resize、center crop 和归一化；TensorRT 执行一次 FP16 engine 推理并打印 Top-5。没有 CUDA 预处理、重复 benchmark 或 CUDA Graph。

## 构建与运行

在 Jetson 上准备 TensorRT、CUDA Toolkit/runtime、OpenCV 4、CMake 和 C++17 编译器。从本目录执行：

```bash
cmake -S . -B build
cmake --build build -j2
./build/mobilenet_demo
```

程序从当前目录读取 `models/mobilenet_v3_small_fp16.engine`、`assets/dog.jpg` 和 `assets/imagenet_classes.txt`。engine 由 `../02_model_convert/04-build_tensorrt.py` 从完整 ONNX 构建；不同设备或 TensorRT 版本需要重新构建。
