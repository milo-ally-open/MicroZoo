# MobileNetV3-Small TensorRT C++ Baseline Demo

该 demo 使用 TensorRT C++ API 加载 FP16 engine，对 `assets/dog.jpg` 完成分类，并分别比较 CUDA 与 OpenCV 预处理路径。

## 构建与运行

在 Jetson 上安装 TensorRT 10、CUDA、CMake、C++17 编译器和 OpenCV 4，然后从本目录运行：

```bash
cmake -S . -B build
cmake --build build -j2
./build/mobilenet_demo
```

运行程序时保持当前目录为 `baseline`，因为 engine、图片和类别文件使用相对路径。

## Demo 内容

- `src/main.cpp`：包含完整 CUDA 预处理测试和 OpenCV 预处理测试；`main()` 只调用这两个测试。
- `kernel/preprocess.cu`：CUDA 图像预处理 kernel 和启动函数。
- `include/preprocess.h`：CUDA 预处理启动接口。
- `include/logger.h`：精简 TensorRT logger。
- `models/mobilenet_v3_small_fp16.engine`：从 `04_model_optimize/baseline/engine/` 复制的 TensorRT engine。
- `assets/dog.jpg`：固定演示图片。
- `assets/imagenet_classes.txt`：Torchvision ImageNet-1K 类别顺序。

OpenCV 在 CPU 上解码 JPEG。之后运行两条路径：OpenCV CPU resize / crop / normalize，再把输入 tensor 传到 GPU；或者先上传 BGR 像素，再由 CUDA kernel 完成双线性缩放、中心裁剪、颜色转换、归一化和 NCHW 排列。两者都采用短边缩放到 256、中心裁剪到 224。

`test_cuda_preprocess()` 和 `test_opencv_preprocess()` 各自完成预处理、TensorRT 推理和 Top-1 分类。`main()` 只调用这两个测试函数。每条路径只打印分类结果、预处理时间、推理时间和端到端时间。

预处理时间包含形成 TensorRT GPU 输入所需的传输：CUDA 路径包括原始像素 H2D 和 CUDA kernel；OpenCV 路径包括 CPU 预处理和 FP32 输入 H2D。推理时间是同一输入预热 10 次后，100 次 TensorRT 调用的平均值，不含 H2D / D2H。端到端时间测量一次预热后的请求，从已解码图片开始，包含预处理、推理、输出 D2H 和 Top-1 计算；不含 JPEG 解码和 engine 初始化。

该 baseline 使用普通 TensorRT `enqueueV3`，不包含 CUDA Graph 或其他优化。engine 的输入输出为 FP32 LINEAR；FP16 指 engine 内部允许使用 FP16 计算。该 engine 针对本机 Jetson、TensorRT 和 CUDA 环境构建，更换目标设备或版本时应重新构建并验证。
