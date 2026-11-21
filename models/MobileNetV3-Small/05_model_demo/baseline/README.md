# MobileNetV3-Small TensorRT C++ Baseline Demo

该 demo 使用 TensorRT C++ API 加载 FP16 engine，对 `assets/dog.jpg` 完成分类并打印 ImageNet Top-5 和推理耗时。

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

OpenCV 在 CPU 上解码 JPEG。程序之后分别运行两条预处理路径：OpenCV CPU 路径，以及先将 BGR 像素复制到 GPU、再由 CUDA kernel 完成双线性缩放、中心裁剪、BGR 转 RGB、ImageNet 归一化和 NCHW 排列的路径。两者都采用短边缩放到 256、中心裁剪到 224。程序会打印两条路径的耗时、输出 tensor 的最大 / 平均绝对差和前 8 个元素。插值实现可能有细微差异。

`test_cuda_preprocess()` 和 `test_opencv_preprocess()` 各自完成图片读取、预处理、engine 加载、TensorRT 推理、计时和 Top-5 输出。前者将 CUDA 输出 tensor 保存下来，后者打印与 CUDA 预处理 tensor 的差异。`main()` 只按顺序调用这两个完整测试流程。

输出 `CUDA preprocess` 耗时包括原始图像 H2D 和 CUDA 预处理 kernel；`OpenCV preprocess` 耗时包括 CPU resize / crop / normalize。差异统计比较归一化后的 FP32 NCHW 输入 tensor。TensorRT 推理耗时先用同一输入预热 10 次，再测量 100 次并输出平均每次时间；计时包括 CPU 提交 TensorRT 推理和等待 GPU 完成，不含预处理、H2D 输入复制和 D2H 输出复制。

该 baseline 使用普通 TensorRT `enqueueV3`，不包含 CUDA Graph 或其他优化。engine 的输入输出为 FP32 LINEAR；FP16 指 engine 内部允许使用 FP16 计算。该 engine 针对本机 Jetson、TensorRT 和 CUDA 环境构建，更换目标设备或版本时应重新构建并验证。
