# YOLO26n V1 优化报告

V1 使用固定输入输出地址的 TensorRT CUDA Graph replay，减少重复推理时逐 kernel 提交的 host 开销。它不改模型权重、精度、engine 或检测后处理。

## 实测结果

设备为 Jetson Orin Nano Super；TensorRT FP16、固定输入 `[1,3,640,640]`、batch 1。计时仅包括 TensorRT engine 执行和每次完成同步，不含图片读取、letterbox、H2D/D2H 或检测框处理。下表为三轮各自统计值的中位数。

| 模式 | Median | P95 | P99 | 变异系数 CV |
|---|---:|---:|---:|---:|
| Eager | 4.452 ms | 7.878 ms | 8.915 ms | 23.6% |
| CUDA Graph | 3.916 ms | 7.391 ms | 8.620 ms | 27.7% |

CUDA Graph 的 median 下降 `12.0%`，P95 下降 `6.2%`。V1 的逐轮 median 为 `3.906–3.918 ms`；eager 为 `4.445–4.474 ms`。长尾依然明显，Graph 的 CV 没有降低，因此这次结果支持减少 enqueue 开销，但不代表测量已稳定。

独立 `trtexec` 重复测量也观察到 GPU median 从 `4.302 ms` 降至 `4.010 ms`，enqueue median 从 `3.270 ms` 降至 `0.021 ms`。原始数据分别见基线 [`summary.json`](../baseline/profiling/trtexec_repeats/summary.json) 和本目录 [`cudagraph_latency.json`](metrics/cudagraph_latency.json)。

## 数值与任务精度回归

脚本对官方 COCO `val2017` 固定 32 张图片逐张比较 eager 与 CUDA Graph 的完整 engine 输出 `[1,300,6]`。32 张全部逐元素相同，最大绝对差为 `0.0`。Graph 只重放同一 engine 命令，因此在该子集上得到相同候选和检测结果；阶段 03 TensorRT FP16 小样本任务指标不变。该子集是流程回归测试，不是完整 COCO 精度验收。

模型 engine 和 I/O buffer 沿用原件。Graph 增加 CUDA graph executable；本次没有单独隔离 graph executable 的进程内存。阶段 03 的 TensorRT GPU allocation 与系统 RAM 采样见 [`validation_report.md`](../../03_model_validate/validation_report.md)，不要将系统总 RAM 当作单个 engine 的进程占用。

## 结论

V1 对固定 shape 的连续请求有效降低了 median 和 P95 延迟，且 32 张回归样本的输出完全相同。收益来自运行时提交路径，不是自定义 CUDA kernel；当前 CV 和 P95 长尾仍高。把 CUDA Graph 集成到最终 C++ 摄像头循环前，应再对包含预处理、推理、NMS 和结果回传的完整链路做多轮 A/B 测量。

## 复现

```bash
cd models/YOLO26n-Detection/04_model_optimize/v1
python3 01-bench-cudagraph.py
```

脚本会在 V1 目录创建 engine 和 metrics 输出；engine 由转换阶段复制，图片及 32 张固定子集由训练阶段准备。
