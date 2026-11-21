# YOLO26n-cls V1 优化报告

## 优化选择

Baseline layer profile 的最大单层中位耗时约 0.072 ms，后续卷积层分布在约 0.02–0.04 ms；没有单个 layer 明显占据整次推理的大部分时间。`trtexec` 提示 eager 路径受 enqueue 时间影响，因此 V1 先验证固定 shape / 固定输入输出地址的 CUDA Graph replay，不写自定义 CUDA 网络算子。

## 测量方法

`01-bench-cudagraph.py` 使用实际 `sample.jpg` 经 `classify_transforms(224)` 预处理后的 tensor，固定同一 engine、同一 context 和地址；每个模式先 warmup 20 次，随后测 200 次，共 3 轮。报告单次 engine 调用的 median、P90、P95、P99、CV。数据不含 JPEG 解码和图像预处理；每次提交后同步以测量请求延迟。脚本同时检查两条路径输出类别和数值。

## 实测结果

此次三轮结果来自 `metrics/cudagraph_latency.json`：

| Run | Eager median (ms) | CUDA Graph median (ms) | Eager CV | Graph CV |
|---:|---:|---:|---:|---:|
| 1 | 1.6396 | 1.1852 | 48.9% | 36.3% |
| 2 | 1.3476 | 0.9133 | 47.6% | 57.7% |
| 3 | 1.2228 | 0.8578 | 43.4% | 41.0% |

三轮中 CUDA Graph median 均低于 eager，单轮降幅约 28–32%。长尾和 CV 仍然较大，不能据此承诺稳定的端到端加速；继续比较时应控制设备温度/负载，保持相同条件并报告分位数。CUDA Graph 不改变模型数值；本次脚本会保存 top1 一致性和输出最大误差。

三轮的 median latency 中位数为 eager `1.3476 ms`、CUDA Graph `0.9133 ms`，降低约 `32.2%`。两条路径 Top-1 一致，最大输出绝对误差为 `0`。各轮 CV 为 `36.3–57.7%`，波动仍很大，因此当前测量结果用于验证方向，不作为稳定 SLA。

V1 C++ demo 同样在固定输入输出地址上 capture / replay TensorRT enqueue，目录为 `05_model_demo/v1/`。C++ 单次 demo 输出只用于确认集成和分类结果；单次 end-to-end 时间不替代多轮基准。

## 复现

```bash
cd models/YOLO26n-CLS/04_model_optimize/v1
python3 01-bench-cudagraph.py
```

运行 C++ V1 demo：

```bash
cd ../../05_model_demo/v1
cmake -S . -B build
cmake --build build -j2
./build/yolo26n_cls_v1_demo
```
