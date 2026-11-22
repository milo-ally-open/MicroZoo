# YOLO26n-cls TensorRT C++ v1 Demo

该 Demo 对固定 `assets/sample.jpg` 做一次图像分类。OpenCV 在 CPU 上完成短边缩放、中心裁剪、RGB 转换和 `[0,1]` 缩放；TensorRT 执行一次 FP16 engine 推理并输出 Top-1/Top-5。没有自定义 CUDA 预处理、重复计时或 CUDA Graph。

## 构建与运行

在 Jetson 上准备 TensorRT、CUDA Toolkit/runtime、OpenCV 4、CMake 和 C++17 编译器，从本目录执行：

```bash
cmake -S . -B build
cmake --build build -j2
./build/yolo26n_cls_demo
```

程序从当前目录读取 `models/yolo26n-cls_fp16.engine`、`assets/sample.jpg` 和 `assets/imagenet_classes.txt`。engine 由 `02_model_convert/04-build_tensorrt.py` 从完整 ONNX 构建；更换 Jetson 或 TensorRT 版本时应重新构建。
