# YOLO26n-cls TensorRT C++ Baseline Demo

在 Jetson 上用 TensorRT C++ API 加载 FP16 engine，对固定 `assets/sample.jpg` 做图像分类。样例来自 `01_model_train/datasets/imagenette2-small/val`，模型输入变换与 Ultralytics `classify_transforms(224)` 对齐。

## 构建和运行

在 Jetson 上安装 TensorRT 10、CUDA、CMake、C++17 编译器和 OpenCV 4，然后从本目录执行：

```bash
cmake -S . -B build
cmake --build build -j2
./build/yolo26n_cls_demo
```

运行时从当前目录读取 `models/yolo26n-cls_fp16.engine`、`assets/sample.jpg` 和 `assets/imagenet_classes.txt`。

## 两条预处理路径

- `test_cuda_preprocess()`：OpenCV 解码 BGR 图像；上传到 GPU，由 `kernel/preprocess.cu` 完成短边缩放到 224、中心裁剪、RGB 转换、像素缩放到 `[0,1]` 和 NCHW 排列；随后运行 TensorRT。
- `test_opencv_preprocess()`：OpenCV 在 CPU 上完成相同的 resize / crop / RGB / `[0,1]` 缩放，再将 FP32 tensor 传到 TensorRT 输入。

`main()` 只调用这两个测试函数。每条路径只打印 Top-1 结果、预处理时间、推理时间和端到端时间。预处理时间包含输入 tensor 到达 GPU 所需的 H2D；推理时间为 100 次 engine 调用平均值，不含传输；端到端时间从已解码图片开始，包含预处理、TensorRT、输出回传和 Top-1 计算，不含 JPEG 解码和 engine 初始化。

Engine 的 I/O 为 FP32，TensorRT 内部允许使用 FP16。该 engine 是在 Jetson Orin Nano Super、JetPack R36.4.3、TensorRT 10.3.0 上构建的；不同设备或软件版本需从 ONNX 重新构建并验证。
