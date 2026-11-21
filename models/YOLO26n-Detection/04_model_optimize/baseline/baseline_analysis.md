# YOLO26n TensorRT 基线剖析与瓶颈分析

## 测量条件

- 设备：Jetson Orin Nano Super；TensorRT 10.3.0；CUDA 12.6。
- Engine：YOLO26n FP16，固定输入 `[1, 3, 640, 640]`，batch 1。
- `trtexec`：预热 500 ms、每轮测量 3 秒、关闭 H2D/D2H；eager 和 CUDA Graph 各重复 3 轮。
- 结果目录：所有 TensorBoard trace、layer profile、Nsight Systems 报告和原始重复测量均保存在 `profiling/`。
- 设备频率没有锁定。TensorRT 报告的 GPU 延迟变异系数为约 25–35%，因此分位数和跨轮趋势比单轮最佳值更有参考意义。

## 基线性能

以下是 `04-repeat_bench.py` 的三轮测量中位数的中位数。每轮的原始数字保存在 [`summary.json`](profiling/trtexec_repeats/summary.json)。

| 执行方式 | GPU latency median | GPU latency P95 | enqueue median | enqueue P95 | throughput | GPU latency CV 中位数 |
|---|---:|---:|---:|---:|---:|---:|
| TensorRT eager | 4.302 ms | 9.068 ms | 3.270 ms | 3.964 ms | 195.5 qps | 32.8% |
| TensorRT CUDA Graph | 4.010 ms | 8.285 ms | 0.021 ms | 0.040 ms | 212.8 qps | 33.7% |

`trtexec` 每轮数据存在明显长尾；CUDA Graph 三轮 GPU median 均在 3.66–4.01 ms，eager 为 4.30–5.03 ms。报告的 P95 仍受长尾影响，不能据此宣称设备延迟已经稳定。

## LayerSummary 与 TensorBoard

`02-bench.py` 为 230 个 TensorRT layer 记录 IProfiler 时间。中位时间靠前的层为：

| TensorRT layer | Median |
|---|---:|
| `/model.1/conv/Conv + PWN(Sigmoid, Mul)` | 0.312 ms |
| TensorRT 融合算子 `ConNegExpAddDivConSliSubSliAddConMulConTraSliMaxSli` | 0.266 ms |
| 输入到首层卷积的 Reformat CopyNode | 0.185 ms |
| `/model.3/conv/Conv + PWN(Sigmoid, Mul)` | 0.185 ms |
| `/model.2/cv2/conv/Conv + PWN(Sigmoid, Mul)` | 0.179 ms |

完整结果见 [`layer_times.json`](profiling/layer_times.json)；TensorBoard layer run 名称为 `TensorRT_Bench`。TensorBoard 的 trace run 使用 `PyTorch_Model` 和 `TensorRT_Engine`。IProfiler 会增加测量开销，layer 时间不能相加作为真实端到端延迟。

没有单个 layer 显著支配整次推理。前几项包含常见卷积/激活融合、张量布局转换和 TensorRT 融合节点；现有证据不支持先为某一个层编写自定义 CUDA kernel。

## Nsight Systems 观察

`profiling/nsys/` 中保存了 eager 与 CUDA Graph 的 `.nsys-rep`。两条 trace 均通过 `nsys stats` 汇总：eager trace 记录了大量逐 kernel launch API；Graph trace 将重复执行表现为 `cudaGraphLaunch`（770 次）。这与 `trtexec` enqueue median 从 3.270 ms 降至 0.021 ms 的结果相符，说明 eager 路径的主机提交开销值得优先处理。

Nsight trace 用于理解 API/Kernel 提交关系，不用来比较绝对延迟；启用 profiler 会改变运行时开销。CUDA kernel 汇总中同时有 Tensor Core 卷积、布局转换与多个融合 kernel，尚无稳定证据证明内存带宽或某个自定义 kernel 是唯一瓶颈。

## 瓶颈判断与建议

1. **优先处理 host enqueue。** 对固定 shape 和固定地址的连续推理使用 CUDA Graph；V1 对照见 [`../v1/optimization_report.md`](../v1/optimization_report.md)。
2. **先保留 TensorRT 已融合的卷积与算子。** 当前 layer 时间分散，继续定制单个 kernel 的收益可能小于测量噪声。
3. **之后检查应用级同步。** 实时 demo 需要逐帧取得框用于绘制；若吞吐仍不够，可测量预处理、推理、后处理之间的 stream 同步和帧缓冲，评估流水线化。
4. **把分辨率调整作为精度/性能实验。** 任何输入分辨率或阈值变化都要重新跑 32 张固定 COCO 子集，并在完整应用数据上复核。
5. **控制测量条件。** 复测时记录电源模式、温度、频率及后台负载；不要只报告单轮最小延迟。当前的 CV 仍高，V1 虽然减少提交开销，但没有消除 GPU 延迟长尾。

## 复现

```bash
cd models/YOLO26n-Detection/04_model_optimize/baseline
python3 01-compare.py
python3 02-bench.py
bash 03-nsight.sh
python3 04-repeat_bench.py
```

TensorBoard：`tensorboard --logdir=profiling`。Nsight Systems GUI 可打开 `profiling/nsys/yolo26n-eager.nsys-rep` 与 `profiling/nsys/yolo26n-cudagraph.nsys-rep`。
