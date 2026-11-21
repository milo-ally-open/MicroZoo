# MobileNetV3-Small TensorRT C++ V1 Demo

V1 使用与 [`../baseline`](../baseline) 相同的 TensorRT FP16 engine、`dog.jpg`、CUDA 预处理 kernel 和 OpenCV CPU 预处理路径。每个测试函数先执行 10 次 warmup，再 capture TensorRT engine 推理并用 CUDA Graph replay 计时和计算分类结果。Graph 固定 TensorRT context 与输入输出地址；图像预处理及 H2D / D2H 不放入 graph。

## 构建与运行

在 Jetson 上安装 TensorRT 10、CUDA、CMake、C++17 编译器和 OpenCV 4，然后从本目录执行：

```bash
cmake -S . -B build
cmake --build build -j2
./build/mobilenet_demo_v1
```

运行时保持当前目录为 `v1`，程序会读取 `models/mobilenet_v3_small_fp16.engine`、`assets/dog.jpg` 和 ImageNet-1K 类别文件。

输出仅包含 Top-1 分类结果、预处理时间、100 次 graph replay 平均时间和单次端到端时间。baseline 与 V1 应在相同设备状态下比较；单次 demo 结果用于确认功能，不替代重复 benchmark。

本机的实际运行数据见 [`validation_report.md`](validation_report.md)。
