# MobileNetV3-Small VALIDATION

## 配置

- 目标设备：NVIDIA Jetson Orin Nano Super Engineering Reference Developer Kit。
- 验证图像：本目录中的 `dog.jpg`；训练目录 `../01_model_train/dog.jpg` 保存相同副本。
- 图像预处理：Torchvision `MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms()`。
- 输入：FP32，NCHW，`[1, 3, 224, 224]`。
- 对照运行时：ONNX Runtime CPU 与 TensorRT FP16。
- 输出：ImageNet-1K logits，`[1, 1000]`。
- 数值与任务验证均使用本地图片，不在脚本中下载文件。
- 本次不测推理速度或系统峰值内存。

验证脚本不加载 PyTorch 模型，也不读取训练 checkpoint；Torch/Torchvision 仅用于图像预处理和 TensorRT 的 CUDA 缓冲区。

## 数值验证：ONNX Runtime CPU vs TensorRT FP16

| MAE | 最大绝对误差 | 平均相对误差 | 余弦相似度 |
|---:|---:|---:|---:|
| 0.050244149 | 0.20483971 | 0.033012241 | 0.99994946 |

## 单图任务验证

样例参考标签为 Samoyed。

| 后端 | Top-1 | 标签匹配 | Top-5 |
|---|---|---|---|
| ONNX Runtime CPU | Samoyed | 是 | Samoyed, Arctic fox, Pomeranian, wallaby, Great Pyrenees |
| TensorRT FP16 | Samoyed | 是 | Samoyed, Pomeranian, Arctic fox, wallaby, Great Pyrenees |

两个后端的 Top-1 一致；Top-5 中 Arctic fox 与 Pomeranian 的排名互换。

## 范围

这次只验证一张样例图，不能代表独立测试集上的分类准确率，也不提供系统内存或稳定性结论。
