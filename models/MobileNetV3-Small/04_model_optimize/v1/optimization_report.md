# MobileNetV3-Small V1 CUDA Graph 优化报告

## 优化内容

V1 在固定 batch 1、输入 `[1,3,224,224]`、固定 TensorRT execution context 和固定输入输出地址下，将 engine 的 `execute_async_v3` capture 成 CUDA Graph，再用 graph replay 提交推理。输入图片使用与 baseline 相同的 `dog.jpg` 和 Torchvision 预处理。预处理、H2D / D2H 和模型本身均未改变；本次只验证减少 CPU 提交开销，不替换 TensorRT layer kernel。

## 复现方式

```bash
cd models/MobileNetV3-Small/04_model_optimize/v1
python3 01-bench-cudagraph.py
```

脚本使用 baseline 目录中的 FP16 engine 和图片，若本目录还没有 engine 会先复制一份。每种提交方式先 warmup 20 次，再测 200 次，共 3 轮；记录 mean、median、P90、P95、P99 和 CV。数值包含 Python 提交与同步开销，不含图像预处理或数据传输。脚本最后检查 eager / graph 的 Top-1 一致性和输出最大绝对误差。原始结果保存在 `metrics/cudagraph_latency.json`。

## 结果与分析

目标 Jetson 实测结果如下。表格中的 median、P95 和 CV 是三轮对应值；汇总行给出三轮结果的中位数。

| Run | Eager median (ms) | CUDA Graph median (ms) | Eager P95 (ms) | Graph P95 (ms) | Eager CV | Graph CV |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.4939 | 1.1782 | 2.7249 | 3.5595 | 35.8% | 54.0% |
| 2 | 1.1436 | 0.9315 | 3.1625 | 2.2627 | 50.6% | 51.9% |
| 3 | 1.1005 | 0.9296 | 2.4796 | 2.3787 | 51.5% | 73.3% |
| 三轮中位数 | 1.1436 | 0.9315 | 2.7249 | 2.3787 | 50.6% | 54.0% |

CUDA Graph 的三轮 median 都比 eager 低；三轮 median 的中位数下降约 `18.6%`。Top-1 一致，eager 与 graph 输出最大绝对误差为 `0`。但 P95 与 CV 显示长尾和运行波动仍明显，尤其 graph CV 的中位数约 `54.0%`。因此这次实测证明优化方向有效，不代表稳定的端到端 SLA。比较之前 baseline `trtexec` 的数据时要注意：baseline `trtexec` 使用默认输入且不含 H2D / D2H；本脚本使用真实 `dog.jpg` 预处理 tensor，并通过 Python 每次同步测量，两组绝对延迟不应直接横向对比。

TensorRT 反序列化 engine 时输出了设备兼容性提示，但 engine 能成功加载并完成所有推理。该 plan 应只在原构建设备和软件栈上使用；跨设备时请从 ONNX 重新构建。

## 精度回归

V1 复用同一个 TensorRT FP16 engine，不改变模型数值路径。`cudagraph_latency.json` 保存 eager / graph 输出 Top-1 一致性和最大绝对误差；分类任务精度基线见 `../baseline/baseline_analysis.md` 与 `03_model_validate/validation_report.md`。
