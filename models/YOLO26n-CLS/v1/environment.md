# YOLO26n-cls 目标环境

## 硬件与系统

- 设备：NVIDIA Jetson Orin Nano Super（8 GB unified memory，Compute Capability 8.7）
- Jetson Linux / JetPack：R36.4.3
- CUDA：12.6.68
- TensorRT：10.3.0
- OpenCV：4.10.0

## Python 环境

- Python：3.10.12（系统 Python）
- PyTorch：`2.5.0a0+872d972e41.nv24.08`（NVIDIA Jetson wheel，CUDA 可用）
- Ultralytics：8.4.155
- ONNX：1.17.0
- ONNX Runtime：1.23.0
- torchvision：随 Jetson PyTorch 环境提供

## C++ 工具

- CMake：3.22.1
- C++：17
- CUDA compiler / TensorRT / OpenCV：使用 JetPack 系统安装版本

此环境用于导出 ONNX、在目标设备构建 engine、运行验证和编译 C++ demo。TensorRT engine 不承诺跨 Jetson、TensorRT 或 CUDA 版本移植；在不同环境部署时应从 ONNX 重新构建并重新验证。
