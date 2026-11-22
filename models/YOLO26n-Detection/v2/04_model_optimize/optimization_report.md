# YOLO26n Detection v2 — 阶段 04 性能剖析记录

## 方法

- 设备：Jetson Orin Nano Super，CUDA 12.6，TensorRT 10.7.0。
- 固定使用一张留出集图片，输入 batch 1、640×640。TensorBoard PyTorch Profiler 分别生成 `PyTorch_FP32` 和 `TensorRT_INT8` worker trace。
- TensorRT `IProfiler` 对已融合 engine 记录 10 次 warmup、10 次 layer profiling 的均值，共得到 203 个 layer 项。逐层采集会增加执行开销，不将这些数字当成正式端到端 latency。
- trace 与 layer JSON 位于本地 `profiling/`，由 `.gitignore` 排除。报告记录本次结论；复现命令见 [`README.md`](README.md)。

TensorBoard 打开后进入 **PYTORCH_PROFILER** 页面，在 Runs 中选择 `profiling`；切换左侧 Workers 的 `PyTorch_FP32` 和 `TensorRT_INT8` 即可看同一 profiler 插件中的两份 trace。若在远程设备上用浏览器访问，启动命令可用 `tensorboard --logdir=profiling --bind_all`，然后打开设备地址的 6006 端口。

本次 trace 检查结果：PyTorch trace 含 CPU operator 事件，TensorRT trace 有逐步 inference 标注；两者均没有 CUDA kernel/activity 事件，CUPTI 权限/设备配置未允许采集 GPU activity。因此目前 TensorBoard 可比较每步标注和 PyTorch CPU operator，不能用这两份 trace 查看 GPU kernel 时间。阶段 04 的 TensorRT GPU 融合层耗时来自 `IProfiler` 的 `tensorrt_int8_layers.json`。本机已有 `sudo -E python3 01-profile.py` 的复现命令；获得 CUPTI 权限后重新生成 trace，再检查 GPU activity 是否出现。

## TensorRT INT8 layer profile

下表是 TensorRT layer profile 中本次均值最高的 7 项：

| Rank | TensorRT layer（节选） | 平均时间 |
|---:|---|---:|
| 1 | `__myl_Top_myl197_2`（TensorRT 生成的内部层名） | 0.7122 ms |
| 2 | `/model.13/.../cv1/conv/Conv` + 融合 Sigmoid/Mul | 0.6826 ms |
| 3 | `/model.10/m/m.0/attn/Softmax` | 0.3216 ms |
| 4 | `/model.13/.../cv2/conv/Conv` + 融合激活 / Add | 0.3095 ms |
| 5 | `/model.23/.../one2one_cv3.1.../conv/Conv` + 融合激活 | 0.2902 ms |
| 6 | `__myl_ResResResConNegExpAddDivResResResConSliSubSliAddConMulConTraSliMaxSli_myl197_1` | 0.2857 ms |
| 7 | 输入到 `/model.0/conv/Conv` 前的 `Reformatting CopyNode` | 0.2428 ms |

完整的 203 项均值保存在 `profiling/tensorrt_int8_layers.json`。`__myl_*` 和融合层名称是 TensorRT 构图/融合后的名字，不能直接当作一个源 ONNX 算子或自定义 CUDA kernel。单层时间也不能简单相加来推断端到端时间。

## 观察与下一步建议

1. 先用 Netron 的 ONNX 路径名和 TensorRT layer profile 对照 `/model.13` 的计算块、`/model.10` 的 attention Softmax，以及 detector head `/model.23`；对 `__myl_*` 先结合 TensorBoard CUDA trace 或 Nsight Systems 找到其调用上下文，不要猜它代表哪个算子。
2. `Reformatting CopyNode` 出现在模型输入端，说明布局/格式转换有可见开销。下一轮应检查 engine 输入格式、预处理输出布局与首层 tactic，确保改动后仍保持同一数值验证条件。
3. 先检查高耗时 Conv 的 TensorRT 精度选择与 tactic，再评估 calibration/图结构；只有确认某段无法被 TensorRT 高效融合且占比仍高时，才考虑自定义 plugin/kernel。
4. 当前 PTQ engine 的小样本 mAP50 比 PyTorch FP32 低约 0.083，先解决精度回归，再比较不同剪枝率或量化配置的性能。性能优化结果应以无 profiler 的独立、重复 latency benchmark 报告。

此阶段只完成数据采集和瓶颈定位，没有改写 TensorRT kernel，也没有宣称 profiler 本身带来性能提升。
