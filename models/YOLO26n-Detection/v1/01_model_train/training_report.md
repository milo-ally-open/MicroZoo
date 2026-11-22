# YOLO26n — 阶段 01 模型与数据报告

## 任务定义

- 任务：COCO 80 类目标检测。
- 模型：Ultralytics 官方 YOLO26n COCO 预训练权重 `yolo26n.pt`。
- 输入：RGB，`[1, 3, 640, 640]`；保持纵横比的 letterbox，padding 值 114，通道转 RGB，像素缩放到 `[0,1]`。
- 原始网络输出：`[1, 84, 8400]` 原始检测预测；部署后处理由 Ultralytics validator 或 demo 的 NMS 完成。
- 目标设备：Jetson Orin Nano Super。
- 训练状态：使用官方预训练权重，**没有针对本项目数据进行微调**。

## 数据集

测试集准备脚本 `00-prepare_coco_val.py` 从 COCO 官方 `val2017` 标注中按固定随机种子 `20261121` 选择 32 张有非 crowd 边界框的图片，并下载对应的 `val2017` 图片。脚本将 COCO 类别 ID 映射到连续的 0–79 类索引，并生成 YOLO 格式标注、`data.yaml` 和图片 ID 清单。实际选择记录在 `datasets/coco-val-32/selection.txt`。该数据集仅用于端到端流程和转换回归；样本数量不足以代表完整 COCO 基准或可靠估算完整验证集 mAP。

## 模型规模与部署前内存评估

从 checkpoint 加载的模型有 2,572,280 个参数，checkpoint 文件为 5.29 MiB。按全部参数估算，FP32 权重约 9.81 MiB，FP16 权重约 4.91 MiB；此静态估算不包含激活、CUDA/TensorRT runtime、workspace、输入输出缓冲区和系统开销。

单图 PyTorch CUDA 推理使用 batch 1、640×640。一次预热后 20 次 Ultralytics `predict` 平均约 40.52 ms（包含图片前处理、模型推理和后处理）；CUDA 峰值分配 57.69 MiB、峰值预留 66.00 MiB，进程最大 RSS 约 1004 MiB。此数据是 Jetson 统一内存环境下的阶段 01 实测，不应与离散显卡显存峰值直接等同。阶段 03 会在完整验证时同时记录系统内存与各后端 CUDA 分配。

## 任务精度

PyTorch FP32、batch 1、640×640 在固定 32 张 `val2017` 图片上的结果：Precision `0.617`、Recall `0.620`、mAP50 `0.638`、mAP50-95 `0.484`。验证器报告每图预处理约 `1.1 ms`、网络推理约 `66.8 ms`、后处理约 `5.5 ms`；CUDA 峰值分配 `64.70 MiB`，峰值预留 `88.00 MiB`。完整结果保存在 `metrics/coco_val_32_pytorch.json`。

这些数据只是 32 张固定 COCO val 图片上的流程验证，类别覆盖和样本量都有限，不能代表完整 COCO val2017、不能作为项目验收精度，也不是微调后精度。

## 运行

```bash
python3 01-download_weights.py
python3 00-prepare_coco_val.py
python3 02-run_model.py
python3 03-evaluate_coco_val.py
```
