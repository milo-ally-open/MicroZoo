# MobileNetV3-Small C++ Demo

使用 TensorRT FP16 engine 和 V1 CUDA Graph 完成单张图片分类。程序包含图像读取、预处理、GPU 输入传输、CUDA Graph 推理、输出传输和 Top-5 分类结果。

## 文件

- `models/mobilenet_v3_small_fp16.engine`：部署用 TensorRT engine。
- `assets/dog.jpg`：默认测试图像。
- `assets/imagenet_classes.txt`：与 ImageNet-1K 输出索引对应的 1000 个类别。
- `src/main.cpp`：完整的 C++ 推理演示流程。
- `include/logger.h`：TensorRT 日志器。

## 环境

在生成该 engine 的 Jetson 上构建和运行。需要 CUDA、TensorRT 10、CMake、C++17 和 OpenCV 4。
TensorRT engine 与生成它的设备、TensorRT/CUDA 运行时版本相关；如更换设备或运行时版本，应在目标设备重新生成 engine。

## 构建与运行

从本目录执行：

```bash
./scripts/build.sh
./build/mobilenet_demo
```

指定其他输入图片：

```bash
./build/mobilenet_demo /path/to/image.jpg
```

也可以依次指定图片、engine 和类别文件：

```bash
./build/mobilenet_demo /path/to/image.jpg models/mobilenet_v3_small_fp16.engine assets/imagenet_classes.txt
```

默认路径相对于本目录。程序输出 Top-5 类别和 softmax 概率，并报告 engine/graph 初始化、图像读取、预处理、GPU 数据传输与 Graph 推理、后处理及单次请求总耗时。

## 预处理和 CUDA Graph

预处理沿用 MobileNetV3-Small ImageNet 权重配置：短边缩放到 256、中心裁剪到 224、BGR 转 RGB、像素缩放到 `[0, 1]`，再使用 ImageNet 均值和标准差归一化，输出 NCHW FP32。图像缩放由 OpenCV `INTER_LINEAR` 完成；与训练验证脚本的 Pillow 插值实现接近，但不保证逐像素一致，少数低排名类别可能不同。

程序预热 TensorRT context 后捕获 `IExecutionContext::enqueueV3`，之后对输入图片执行 H2D 传输、CUDA Graph replay 和 D2H 输出读取。H2D/D2H 与图像预处理位于 Graph 外部；GPU 推理部分使用 V1 的 Graph replay。engine 的外部输入和输出为 FP32，engine 内部使用 FP16 计算。

`request_total` 统计 engine 初始化完成后的一次请求，从读取图片开始，到输出 Top-5 结束；不包含 engine 反序列化和 graph capture。`setup` 单独记录这部分初始化耗时。它是演示程序的单次端到端时间，不等同于 V1 中排除了预处理和数据传输的推理延迟。
