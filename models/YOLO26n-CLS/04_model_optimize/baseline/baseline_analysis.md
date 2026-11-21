# YOLO26n-cls 基线性能剖析

## 固定测试条件

- 目标设备：Jetson Orin Nano Super；Jetson Linux R36.4.3，CUDA 12.6，TensorRT 10.3.0。
- Engine：本机从 `02_model_convert/onnx/yolo26n-cls.onnx` 构建的 FP16 engine；静态 batch 1、输入 224×224。
- Profiler 输入：`sample.jpg`，同一个 `classify_transforms(224)` 预处理结果。
- `01-compare.py`：PyTorch CUDA FP16 与 TensorRT engine 都在 GPU 上运行；TensorBoard trace 用于观察 kernel 和调度，不作为正式延迟基准。
- `02-bench.py`：真实预处理后的图像常驻 GPU，只 profile TensorRT engine，导出逐层时间到 TensorBoard。
- `03-nsight.sh`：用 `trtexec` 采集 eager / CUDA Graph 以及 Nsight Systems 时间线。

## 实测结果

`01-compare.py` 生成 `PyTorch_Model` 和 `TensorRT_Engine` 两组 TensorBoard trace。`02-bench.py` 使用真实 `sample.jpg` 的预处理结果，对 98 个 TensorRT layer 采样并输出 `profiling/layer_times.json`。逐层数据中耗时最高的是分类头 Gemm，中位数 `0.072128 ms`、平均 `0.081102 ms`；第二高的融合卷积层中位数为 `0.037344 ms`。没有单个 layer 占据整个推理时长的大部分，不能仅凭该表认定某一个算子是主要瓶颈。

`04-repeat_bench.py` 对 TensorRT engine 做三次独立的、无 profiler 的 `trtexec` 测量。下表中的 latency / P95 / CV 取三次测量结果的中位数，throughput 为三次测量吞吐量的中位数。

| 提交方式 | Median latency (ms) | P95 (ms) | Throughput (qps) | GPU latency CV |
|---|---:|---:|---:|---:|
| Eager | 0.9154 | 2.2915 | 926.4 | 53.3% |
| CUDA Graph | 0.6189 | 1.3831 | 1391.2 | 63.2% |

该组 `trtexec` 测量不含输入输出传输。CUDA Graph 的三轮 median latency 中位数比 eager 低约 32.4%，吞吐量中位数高约 50.1%；但 CUDA Graph 的 CV 仍高且更大，长尾也明显，当前数据只能说明有优化潜力，不能作为稳定端侧延迟承诺。三轮原始日志保存在 `profiling/trtexec_repeats/`。

Nsight Systems 记录保存在 `profiling/nsys/yolo26n-cls-eager.nsys-rep` 和 `profiling/nsys/yolo26n-cls-cudagraph.nsys-rep`。三轮 `trtexec` enqueue latency 中位数的中位数：eager 约 `0.706 ms`、CUDA Graph 约 `0.012 ms`，说明主机提交开销值得继续观察。逐层 TensorBoard 数据和完整 trace 都位于 `profiling/`。

Profiler 会增加测量开销；层耗时不能简单相加当作端到端延迟。Latency 与吞吐应使用不挂 profiler 的重复测量记录。测试集只有 100 张，它用于精度回归，不用于性能吞吐结论。测量期间没有锁定设备频率；报告中保留了各轮分位数和 CV，方便后续按相同条件复测。

## 优化判断

基于本次 layer profile 与 Nsight 结果，先把 CUDA Graph 作为运行时优化路线；V1 已在同一 engine、同一图片上测量并验证输出。单层中位数最高的 Gemm 只有约 `0.072 ms`，暂时没有证据支持先写自定义 CUDA layer kernel。下一步应在相同设备状态下扩大重复轮次，并同步记录温度、功耗和系统负载，确认 Graph 的长尾是否可接受。

## 复现

```bash
cd models/YOLO26n-CLS/04_model_optimize/baseline
python3 01-compare.py
python3 02-bench.py
bash 03-nsight.sh
python3 04-repeat_bench.py
tensorboard --logdir=profiling
```
