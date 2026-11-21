# YOLO26n 检测环境记录

本次运行目标设备是 Jetson Orin Nano Super。

| 组件 | 版本 / 信息 |
|---|---|
| Jetson Linux | R36.4.3（本机已有部署记录） |
| Python | 3.10.12，`/usr/bin/python3` |
| PyTorch | 2.5.0a0+872d972e41.nv24.08 |
| CUDA | 12.6，`torch.cuda.is_available() == True` |
| GPU | Orin |
| Ultralytics | 8.4.155 |
| ONNX | 1.17.0 |
| ONNX Runtime | 1.23.0 |
| TensorRT | 10.3.0 |
| OpenCV | 4.10.0 |
| CMake | 3.22（系统工具链） |

权重来自 Ultralytics 官方 assets release `v8.4.0` 的 `yolo26n.pt`。脚本固定 Ultralytics 版本，保留 JetPack 中的 PyTorch / TensorRT 组合，不通过 requirements 文件重新安装框架。
