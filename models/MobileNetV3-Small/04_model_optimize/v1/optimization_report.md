# V1：TensorRT CUDA Graph

## 目标与改动

将现有 TensorRT engine 的 `execute_async_v3` 捕获到 CUDA Graph，再比较普通 TensorRT enqueue 和 Graph replay 的单样本延迟。Graph 重用固定输入、输出缓冲区和执行上下文，减少每次推理的主机端提交开销；模型结构、权重和 engine 没有改变。

脚本：`01-cuda_graph_latency.py`

## 测量设置

- 设备：Jetson Orin Nano Super；GPU 时钟使用设备动态设置，没有锁钟。
- Engine：`../baseline/engine/mobilenet_v3_small_fp16.engine`
- 输入：`../baseline/dog.jpg`，batch 1，`[1, 3, 224, 224]`
- 每次运行预热 50 次，5 轮，每种模式每轮 100 次，共每种模式 500 次。
- 每轮先测 100 次 eager，再测 100 次 CUDA Graph。
- 两种模式共享 TensorRT context 和输入、输出缓冲区；每次计时前后同步 CUDA。
- 计时不包括图像预处理及输入传输，包含主机端调用和同步等待。因此这是同步单次推理延迟，不是纯 GPU kernel 时间或吞吐率。
- 每次运行前检查零输入和狗图输入的 CUDA Graph 输出与普通 TensorRT 输出一致。

## 结果

每次为单独启动脚本测得的数据；P50 变化按同次运行计算。

| 运行 | 模式 | Mean (ms) | P50 (ms) | P95 (ms) | P99 (ms) | P50 变化 |
|---|---|---:|---:|---:|---:|---:|
| 1 | TensorRT eager | 1.726 | 1.438 | 2.979 | 4.734 | — |
| 1 | CUDA Graph | 1.306 | 1.043 | 2.463 | 4.078 | -27.5% |
| 2 | TensorRT eager | 1.671 | 1.285 | 3.512 | 4.765 | — |
| 2 | CUDA Graph | 1.123 | 0.934 | 2.381 | 3.579 | -27.3% |
| 3 | TensorRT eager | 1.464 | 1.281 | 2.762 | 4.239 | — |
| 3 | CUDA Graph | 1.033 | 0.918 | 2.010 | 3.873 | -28.3% |

三次运行中，CUDA Graph 的 P50 都低于普通 enqueue，平均 P50 从 1.335 ms 降到 0.965 ms，约降低 27.7%。高分位数仍显示长尾，动态时钟和设备负载会影响单次延迟；因此这里报告三次结果及分位数，不把 P50 改善解读成所有场景都能获得同等收益。

逐次原始数据：

- `01-cuda-graph-latency-run1.csv`
- `01-cuda-graph-latency-run2.csv`
- `01-cuda-graph-latency-run3.csv`
- `01-cuda-graph-latency.csv` 是最近一次运行的数据。

## 结论与下一步

CUDA Graph 是本模型的有效 V1 执行优化：在当前固定 shape、batch 1 的设备测量中，稳定降低了 P50 延迟。适用于输入 shape 固定且能复用缓冲区的推理流程；若应用的瓶颈主要来自 GPU 计算而不是主机端提交，收益可能更小。Graph 目前只在此基准脚本中使用，还没有接入正式 demo。

下一步可以做 V2 瓶颈优化：现有 baseline layer profile 将输入 reformat（约 5.1%）和分类器尾部的两个层（约 4.5%、3.8%）标为候选。但现有 engine 的层信息不够详细，不能仅凭这些数字直接改 kernel。先导出带详细层信息的独立 engine profile，核对相关层的类型、shape、精度和融合边界，再决定是否调整输入格式或分类器尾部，并在相同条件下比较精度与延迟。
