# YOLO26n Detection v2 — 阶段 01 模型与数据准备

v2 不训练、不微调。输入权重是 v1 使用的 Ultralytics COCO 预训练 `yolo26n.pt`；转换阶段会从该 checkpoint 重新导出 v2 自己的 FP32 ONNX。

`01-prepare-calibration.py` 使用 v1 固定的 COCO val32 清单和随机种子 `20261122`，拆出 24 张校准图片与 8 张独立留出图片。校准集只用于 TensorRT PTQ；留出集只用于剪枝/量化后的验证。复现命令见本目录 `README.md`。

这 8 张留出图是流程回归样本，不足以代表完整 COCO 检测精度。正式部署前应扩大校准集并使用独立测试集重新评估。

| 项目 | 设置 |
|---|---|
| 训练 / 微调 | 不进行 |
| 来源权重 | v1 `yolo26n.pt`，COCO 预训练 |
| PTQ 校准图片 | 固定 24 张，随机种子 `20261122` |
| 留出验证图片 | 固定 8 张，与校准集不重叠 |
| 输入 | RGB NCHW，640×640，FP32 `[0,1]` |

任务指标由阶段 03 对同一 8 张留出图评估原始 PyTorch、剪枝 FP32 ONNX 和 TensorRT INT8 engine；结果写入 `../03_model_validate/metrics/task_accuracy.json`。
