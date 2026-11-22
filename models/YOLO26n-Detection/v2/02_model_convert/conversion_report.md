# YOLO26n Detection v2 — 转换记录

## 方法与边界

- 不训练、不微调。输入是 v1 使用的 Ultralytics COCO 预训练 `yolo26n.pt`，由 `01_export_onnx.py` 在 v2 重新导出完整 FP32 ONNX。
- 输入固定为 batch 1、`[1,3,640,640]`、RGB NCHW、FP32 `[0,1]`；ONNX opset 17，不包含 NMS。
- `02-prune-onnx.py` 对 `/model.7/conv/Conv` 做结构化通道裁剪：按卷积滤波器 L1 范数保留 250/256 个输出通道（移除 6 个，2.34%），并同步裁剪 `/model.8/cv1/conv/Conv` 的输入权重。脚本检查 Conv→Sigmoid/Mul→Conv 的连接关系和权重维度，遇到图结构变化会停止。
- `04-build-tensorrt-int8.py` 使用 TensorRT 10 的 `IInt8EntropyCalibrator2` 做 PTQ；校准数据经本仓库的 Ultralytics LetterBox 预处理。启用 INT8 与 FP16 builder 标志，实际 engine 可按层选择可用精度，不应理解为所有层都必定以 INT8 执行。
- 参考 PPQ YOLO 示例的代表性图片校准思路，但本项目不安装/调用 PPQ，也不生成 Q/DQ ONNX；量化范围由 TensorRT builder 在构建时计算。

## 数据与环境

`01_model_train/01-prepare-calibration.py` 从固定 COCO val32 清单按随机种子 `20261122` 拆分 24 张校准图和 8 张独立留出图。校准输入是 `[24,3,640,640]` FP32 `[0,1]`。该小数据集只用于复现转换链路，不足以证明正式精度。

| 项目 | 实测值 |
|---|---|
| 设备 | Jetson Orin Nano Super |
| TensorRT | 10.7.0 |
| CUDA Toolkit | 12.6 |
| OpenCV | 4.10.0 |
| 校准图片 / batch | 24 / 1 |
| 留出图片 | 8 |

TensorRT 10 已将 implicit INT8/calibrator 接口标记为 deprecated；TensorRT 11 移除了这些接口。当前脚本针对设备上的 TensorRT 10.7。升级到 TensorRT 11 时需要迁移到显式 Q/DQ 量化模型。

## 最终产物

| 文件 | 大小 | 说明 |
|---|---:|---|
| `onnx/yolo26n_fp32.onnx` | 9,893,947 B（9.436 MiB） | v2 从 checkpoint 导出的 FP32 ONNX |
| `onnx/yolo26n_pruned.onnx` | 9,889,830 B（9.432 MiB） | 裁剪后 FP32 ONNX，比原图小 4,117 B（约 0.04%） |
| `engine/yolo26n_pruned_int8.engine` | 5,835,892 B（5.566 MiB） | TensorRT 10.7.0 在目标 Jetson 上构建的量化 engine |

本次只移除一个卷积的 6 个通道，ONNX 文件大小几乎没有变化；当前主要的文件体积收益来自 TensorRT engine 的低精度构建，而不是这次保守剪枝。engine 构建成功，耗时约 946 秒。engine I/O 是 `images` FP32 `[1,3,640,640]` → `output0` FP32 `[1,300,6]`；INT8 指 engine 内部执行精度。

TensorRT calibration cache 为 `calibration/yolo26n_int8_entropy.cache`。改变剪枝图时，剪枝脚本会删除旧 cache；若更换校准图片或预处理方法，也必须删除该 cache 后重建。

## 结论

引擎已经可由 TensorRT 10.7 反序列化并执行，数值和任务回归结果见 [`../03_model_validate/validation_report.md`](../03_model_validate/validation_report.md)。在当前 8 张留出集上，剪枝 FP32 ONNX 的 mAP50 为 0.764，TensorRT INT8 engine 为 0.682，而 PyTorch FP32 为 0.765。INT8 相对 PyTorch 的 mAP50 下降约 0.083，因此这个版本完成了可复现的 PTQ/部署链路，但小样本结果尚不能证明精度可接受；正式结论需要更大的代表性校准集和独立验证集。
