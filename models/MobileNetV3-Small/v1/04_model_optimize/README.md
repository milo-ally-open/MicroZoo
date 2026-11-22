# 阶段 04：模型优化

当前阶段保留最基础的 PyTorch 与 TensorRT profiling 对比：使用同一张 `dog.jpg`，分别采集 PyTorch 模型和 TensorRT engine 的 CPU/CUDA trace，并通过 TensorBoard 查看。该脚本用于查看执行过程，不输出延迟基准结论。

## 运行

在目标 Jetson 环境中进入本目录后运行：

```bash
sudo -E python 01-profile.py
tensorboard --logdir=profiling
```

依赖：PyTorch（包含 `torch.profiler`）、`torch-tb-profiler`、`tensorboard`、Torchvision、Pillow 和 TensorRT。运行后，profiling trace 保存在本目录的 `profiling/` 下，该目录为本地生成物，不纳入仓库。

当前不做量化、CUDA Graph、算子优化或自定义 TensorRT Plugin；这些工作留待后续版本。
