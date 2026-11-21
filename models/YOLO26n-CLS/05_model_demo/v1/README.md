# YOLO26n-cls TensorRT C++ V1 Demo

V1 与 `../baseline` 使用相同的 engine、测试图片、CUDA 预处理 kernel 和 OpenCV 预处理路径。每个测试函数先进行 10 次 eager warmup，再使用固定输入输出缓冲区 capture TensorRT engine 执行，并通过 CUDA Graph replay 完成计时和最终分类。

```bash
cmake -S . -B build
cmake --build build -j2
./build/yolo26n_cls_v1_demo
```

输出包含两种预处理路径的 Top-1、预处理耗时、engine 推理耗时和端到端耗时。Graph 只包含 TensorRT 执行，不捕获图像预处理或主机/设备数据传输。比较 baseline 和 V1 时使用同一设备状态与测试图。两者使用同一个 engine；变化仅在运行时提交方式，模型权重和精度保持一致。
