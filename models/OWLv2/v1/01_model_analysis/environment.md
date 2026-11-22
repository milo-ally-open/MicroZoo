# 01 环境记录

## 目标设备

| 项目 | 实测信息 |
|---|---|
| 设备 | NVIDIA Jetson Orin Nano Super |
| 架构 / 系统 | aarch64，Ubuntu 22.04.5，Jetson Linux R36.4.3 |
| 系统 Python | 3.10.12 (`/usr/bin/python3`) |
| NVIDIA PyTorch | `2.5.0a0+872d972e41.nv24.08` |
| CUDA | 12.6；`torch.cuda.is_available()` 为 `True` |
| TensorRT | 10.7.0 |
| GPU | Orin，Compute Capability 8.7 |

## Conda 环境

专用环境名为 `microzoo-owlv2-v1`，Python 3.10.22，NumPy 1.26.4，SciPy 1.15.2，Pillow 10.4.0，Transformers 4.48.3，Hugging Face Hub 0.28.1，Safetensors 0.5.3。

JetPack 的 PyTorch、torchvision 和 OpenCV 装在系统 Python 3.10 下；TensorRT Python 包位于系统 Python 3.10 的 dist-packages。环境创建脚本只将这些目录加入 Conda Python 的搜索路径，并从 Conda Forge 建环境；NumPy、Pillow 使用 Conda 包，SciPy 使用固定的 PyPI 二进制 wheel，避免系统 SciPy 与 Conda NumPy 的版本冲突。没有在 Conda 中安装另一个 Torch。

```bash
cd models/OWLv2/v1/01_model_analysis
bash setup_conda_environment.sh
conda activate microzoo-owlv2-v1
python 01-check_environment.py
```

环境检查已确认 Conda Python 能导入系统 NVIDIA PyTorch、TensorRT，并在 Orin CUDA 设备上分配和计算张量。`01-check_environment.py` 输出的是运行时实际版本；更换 JetPack 后应重新检查并记录，不应假定 engine 或原生扩展兼容。

此环境用于 OWLv2 PyTorch 基线分析。ONNX 导出和 TensorRT engine 构建仍需在后续转换阶段单独验证。
