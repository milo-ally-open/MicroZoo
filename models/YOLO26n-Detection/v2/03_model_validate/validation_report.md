# YOLO26n Detection v2 — 阶段 03 验证报告

## 验证设置

- 数据：COCO val2017 固定 val32 子集按种子 `20261122` 拆分；24 张仅校准，8 张仅验证。
- 输入：640×640、Ultralytics LetterBox（`auto=False, scaleup=True`）、BGR→RGB、NCHW、FP32 `[0,1]`、batch 1。
- 任务评估：同一 Ultralytics detection validator 分别运行 PyTorch FP32、原始 FP32 ONNX、剪枝 FP32 ONNX 和剪枝 TensorRT INT8 engine。
- 数值评估：用留出集首张图，分别比较原始 ONNX→剪枝 ONNX、剪枝 ONNX→TensorRT INT8。候选阈值 0.001，同类别 IoU 匹配阈值 0.5。
- 设备：Jetson Orin Nano Super；TensorRT 10.7.0。ONNX Runtime 使用 CPUExecutionProvider。

## 任务精度

| 后端 | P | R | mAP50 | mAP50-95 | validator inference ms/image |
|---|---:|---:|---:|---:|---:|
| PyTorch FP32，CUDA | 0.6338 | 0.7286 | 0.7646 | 0.5334 | 153.36 |
| 原始 FP32 ONNX，CPU | 0.6732 | 0.6952 | 0.8025 | 0.5943 | 231.05 |
| 剪枝 FP32 ONNX，CPU | 0.6876 | 0.6620 | 0.7636 | 0.5273 | 369.65 |
| TensorRT INT8，CUDA | 0.5254 | 0.6883 | 0.6820 | 0.4798 | 8.06 |

相对 PyTorch FP32，TensorRT INT8 在这 8 张图上的 mAP50 低 0.0826、mAP50-95 低 0.0536。当前 INT8 精度有明显下降，不能据此宣称精度达标。原始/剪枝 ONNX 在 CPU 上运行，PyTorch/TensorRT 在 GPU 上运行，表中延迟包含 validator 计时行为且运行设备不同，不作为严格性能对比；正式延迟应在同一设备、相同预处理和重复测量条件下单独测量。

这是 8 张图的流程回归，不是 COCO 正式 benchmark；样本少会导致指标波动。正式验收需要扩大独立验证集，也需要让校准集覆盖摄像头实际场景。当前没有进行精度恢复微调。

## 输出数值与候选匹配

TensorRT 成功反序列化并执行，I/O 为 FP32 `images [1,3,640,640]` → `output0 [1,300,6]`。所有三种 ONNX/TRT 后端的输出 shape 均为 `[1,300,6]`。具体指标记录在 [`metrics/numerical.json`](metrics/numerical.json)。

| 对比 | 参考 / 转换候选数 | 同类别 IoU≥0.5 匹配 | 匹配框平均 IoU | 匹配框坐标 MAE | 置信度 MAE |
|---|---:|---:|---:|---:|---:|
| 原始 ONNX → 剪枝 ONNX | 216 / 176 | 125 | 0.9010 | 2.646 px | 0.02445 |
| 剪枝 ONNX → TensorRT INT8 | 176 / 228 | 149 | 0.9132 | 2.047 px | 0.01417 |

候选集合也有未匹配项：第一组 91 个参考、51 个转换候选未匹配；第二组分别为 27 和 79 个。候选按分数排序，剪枝或量化会改变 TopK 顺序，所以逐行 tensor MAE 不代表一一对应框的误差；需结合候选匹配数和 mAP 阅读。单图的平均匹配 IoU 较高也不能抵消 8 图 mAP 下降。

## 内存观察

`03-memory-validation.sh` 在任务验证期间采集 `tegrastats`，14 条采样中系统 RAM 峰值 5,979/7,620 MiB、swap 约 2,269/3,810 MiB、最高 `tj` 温度 56.8°C、最高 VDD_IN 10,560 mW。它们是整个系统在验证期间的读数，不是某个模型单独的内存或功耗。

验证脚本记录的 CUDA allocator 峰值：PyTorch FP32 allocated/reserved 为 63.55/88 MiB；TensorRT INT8 为 47.33/88 MiB。ONNX Runtime 在 CPU 上运行，没有 CUDA allocator 读数。

## 验收判断

- 流程：通过。PyTorch、原始/剪枝 ONNX、TensorRT INT8 均能在各自运行时执行，ONNX 与 engine 输出接口匹配。
- 数值/任务精度：未通过正式精度验收。当前 PTQ engine 在小留出集的 mAP50 比 PyTorch FP32 低约 8.3 个百分点；需先增加校准样本并进一步评估量化误差，或考虑减小剪枝/改用显式 Q/DQ 流程。
- 性能：阶段 03 validator 数字不能作为正式 latency claim；TensorRT INT8 的层级剖析另见阶段 04。
