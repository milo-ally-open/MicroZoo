# YOLO26n-cls — 阶段 01 模型研发记录

## 任务和输入

- 任务：ImageNet-1K 单标签图像分类。
- 模型：Ultralytics YOLO26n-cls 官方预训练 checkpoint。
- 输入：RGB，`[1, 3, 224, 224]`；Ultralytics 分类变换（保持纵横比、短边缩放、中心裁剪、像素缩放到 `[0,1]`，不做 ImageNet mean/std 归一化）。
- 输出：ImageNet-1K 1000 类分数/概率，Top-1 / Top-5。
- 目标设备：NVIDIA Jetson Orin Nano Super，Jetson Linux R36.4.3 / TensorRT 10.3.0。

## 数据和权重

本阶段采用官方 ImageNet-1K 预训练模型作为可部署基线，不在没有项目专属标注数据的情况下伪称完成了微调。权重通过 `01-download_weights.py` 从 Ultralytics assets `v8.4.0` 获取。独立小型任务验证使用 ImageNette2-160 `val` split 中每类按文件名排序的前 10 张图，共 100 张；由 `00-download_imagenette.sh` 从固定 URL 获取，`00-prepare_testset.py` 生成固定子集和清单。`sample.jpg` 是该测试集内 tench 类的第一张图，复制到相关阶段目录用于冒烟、剖析和 demo。该子集不能代表完整 ImageNet-1K 测试集。

类别映射固定为 ImageNet synset 到 ImageNet-1K 索引；代码见 `03-evaluate_imagenette.py`。推理脚本不下载图片或数据，数据下载由单独脚本显式完成。

## 模型规模、内存评估和部署可行性

官方 YOLO26n-cls 配置为 2,812,104 个参数；按 FP32 权重估算约 10.73 MiB，FP16 权重约 5.36 MiB。该估算不包含激活、TensorRT workspace、运行时和输入输出缓冲区。100 张测试集、batch 32 的 PyTorch CUDA 推理峰值分配为 159.3 MiB。目标设备是共享统一内存的 8 GB Jetson Orin Nano Super，阶段 03 验证期间系统 RAM 峰值约 4.87 GiB，且 engine 能成功加载和推理。依据当前模型规模和峰值数据，原始模型的部署内存可行性初步判定为“可行”；仍需为实际应用的其他进程和系统预留内存。

## 运行方法

从仓库根目录执行：

```bash
cd models/YOLO26n-CLS/v1/01_model_train
python3 01-download_weights.py
bash 00-download_imagenette.sh
python3 02-run_model.py
python3 03-evaluate_imagenette.py
```

使用系统 Python；`requirements.txt` 固定 Ultralytics 版本为 8.4.155。Jetson 上不应让 pip 替换 NVIDIA 提供的 PyTorch/CUDA/TensorRT 运行时。此项目使用官方预训练权重，未执行微调；可复现交付物是权重下载、测试集准备、预处理和精度评估脚本。

## 实测结果

本机运行结果：固定 100 张子集上 PyTorch Top-1 为 64%，Top-5 为 82%；batch 32 用时约 1.06 s，PyTorch 峰值 CUDA 分配约 159.3 MiB。该小样本用于流程与转换回归，不代表完整 ImageNet-1K 精度。
