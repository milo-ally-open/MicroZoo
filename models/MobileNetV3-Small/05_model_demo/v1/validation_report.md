# MobileNetV3-Small C++ V1 Demo 验证

## 运行条件

在同一 Jetson 上依次运行 baseline 与 V1，使用相同 FP16 TensorRT engine、`dog.jpg` 和两种预处理路径。V1 的 graph 只包含 TensorRT 推理；图像预处理与输入 / 输出传输仍按 baseline 路径执行。

## 单次 demo 输出

| 版本 | 路径 | Top-1 | 预处理 (ms) | 推理均值 (ms) | 端到端 (ms) |
|---|---|---|---:|---:|---:|
| Baseline | CUDA | Samoyed (73.70%) | 1.945 | 1.307 | 3.958 |
| V1 | CUDA + CUDA Graph | Samoyed (73.70%) | 1.663 | 1.564 | 3.598 |
| Baseline | OpenCV | Samoyed (73.90%) | 3.780 | 1.387 | 4.915 |
| V1 | OpenCV + CUDA Graph | Samoyed (73.90%) | 4.546 | 1.138 | 5.688 |

两条路径的分类结果与 baseline 一致。这里的推理列是每个程序 100 次调用的均值；预处理与端到端各测一次，变化较大。本表用于确认 C++ Graph demo 能加载 engine、得到正确分类结果，不据此推断稳定端到端性能提升。更可重复的 engine-only 延迟比较见 `04_model_optimize/v1/optimization_report.md`。

TensorRT logger 显示 engine plan 的跨设备可移植性提示，但 engine 在本机成功反序列化并完成推理。该 engine 应留在构建设备和对应软件环境使用；迁移到其他 Jetson 时应从 ONNX 重建。
