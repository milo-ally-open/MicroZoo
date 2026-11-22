# MobileNetV3-Small 环境记录

本文区分本地 WSL 开发环境与 Jetson 端侧环境。前者用于编写脚本、下载权重和运行 PyTorch 样例；TensorRT engine 与 C++ Demo 应在 Jetson 目标环境构建或验证。

## 本地开发环境（WSL）

| 项目 | 记录值 |
|---|---|
| 系统 | Ubuntu 24.04.5 LTS / WSL2 |
| Python | 3.14.6（WSL `base` 环境） |
| PyTorch | `2.14.1+cu126` |
| Torchvision | `0.29.1+cu126` |
| CUDA 可用性 | `torch.cuda.is_available()` 为 `True` |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |

以上版本来自本地环境查询。模型脚本使用 Torch、Torchvision、Pillow；ONNX 导出还需要 `onnx`。本地 WSL 的 CUDA/Torch 环境用于开发和样例推理，不代表 Jetson TensorRT 部署环境。

## Jetson 端侧目标环境

| 项目 | 记录值 |
|---|---|
| 设备 | NVIDIA Jetson Orin Nano Super Engineering Reference Developer Kit |
| Jetson Linux | R36.4.3 |
| Python | 3.10.12（系统 Python，目标平台软件栈记录） |
| CUDA | 12.6.68（目标平台软件栈记录） |
| TensorRT | 10.3.0 |
| PyTorch | `2.5.0a0+872d972e41.nv24.08`（NVIDIA Jetson wheel，目标平台软件栈记录） |
| Torchvision | 随 Jetson PyTorch 环境提供 |
| ONNX | 1.17.0（目标平台软件栈记录） |
| ONNX Runtime | 1.23.0（目标平台软件栈记录） |
| OpenCV | 4.10.0（目标平台软件栈记录） |
| CMake / C++ | CMake 3.22.1 / C++17（Demo 构建环境记录） |

MobileNetV3-Small 转换报告确认目标 Jetson、Jetson Linux R36.4.3 和 TensorRT 10.3.0；其余 Jetson 软件包版本沿用仓库同一目标平台的软件栈记录。性能剖析报告只记为 Jetson Orin，具体 SKU 未记录，因此不能据此确认该组 profiling 数值一定来自 Orin Nano Super。

## 依赖与产物边界

- 权重下载和 PyTorch 单图推理：Python、PyTorch、Torchvision、Pillow；单图推理还要求 CUDA 可用。
- ONNX 导出与切分：PyTorch、Torchvision、ONNX。
- 数值验证：NumPy、ONNX Runtime、TensorRT、PyTorch CUDA、Torchvision、Pillow。
- C++ Demo：CUDA Toolkit、TensorRT 10、OpenCV 4、CMake、C++17 编译器。
- TensorRT engine 是针对构建设备及其 TensorRT/CUDA 软件栈生成的运行时产物。换设备或软件版本时，应重新构建并在目标机验证；不要把 WSL GPU 上生成的 engine 当作 Jetson 的可移植模型文件。

本文件记录已观察到的环境，不会自动安装或升级系统软件。复现前请在对应设备检查实际版本。
