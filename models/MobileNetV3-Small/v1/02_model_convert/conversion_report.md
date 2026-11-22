# MobileNetV3-Small 转换报告

## 转换摘要

- 状态：已生成完整 ONNX 模型和 TensorRT engine。
- 转换链路：Torchvision 预训练权重 → ONNX opset 17 → TensorRT FP16 engine。
- 目标设备：NVIDIA Jetson Orin Nano Super Engineering Reference Developer Kit，Jetson Linux R36.4.3。
- TensorRT 版本：10.3.0。
- 主要部署产物：`engine/mobilenet_v3_small_fp16.engine`。

## 模型输入与输出

| 项目 | 规格 |
|---|---|
| ONNX 输入 | `images`，FP32，NCHW，固定形状 `[1, 3, 224, 224]` |
| 输入预处理 | ONNX 图外执行；使用 Torchvision `MobileNet_V3_Small_Weights.IMAGENET1K_V1` 预处理 |
| ONNX 输出 | `logits`，FP32，形状 `[1, 1000]`，ImageNet-1K 原始分类分数 |
| 后处理 | Softmax、Top-K 和类别名称映射在 ONNX 图外执行 |
| 动态形状 | 未配置；导出使用固定 batch 1 和 `224 × 224` 输入 |

## 转换配置

### PyTorch → ONNX

- 导出脚本：`01-export_onnx.py`
- 输入权重：`pt/mobilenet_v3_small-047dcff4.pt`
- ONNX 算子集版本：17
- 常量折叠：启用
- Exporter：TorchScript/legacy exporter（`dynamo=False`）
- 量化/校准：未使用

### ONNX → TensorRT

- 构建脚本：`04-build_tensorrt.py`
- ONNX 解析器：TensorRT ONNX Parser
- 优化精度：启用 FP16 builder flag；未启用 INT8 或校准
- Workspace 上限：512 MiB
- Batch 与分辨率：固定为 `[1, 3, 224, 224]`
- Engine 在目标 Jetson Orin Nano 上构建

FP16 builder flag 允许 TensorRT 为适用层选择 FP16 实现；本报告不将其表述为所有层和 I/O 均为 FP16。

## 产物清单

| 产物 | 路径 | 文件大小 |
|---|---|---:|
| 转换阶段权重副本 | `pt/mobilenet_v3_small-047dcff4.pt` | 10,306,551 bytes（9.83 MiB） |
| 完整 ONNX 模型 | `onnx/mobilenet_v3_small.onnx` | 10,181,045 bytes（9.71 MiB） |
| 完整 TensorRT FP16 engine | `engine/mobilenet_v3_small_fp16.engine` | 6,223,108 bytes（5.93 MiB） |
| Backbone ONNX（实验产物） | `onnx/mobilenet_v3_small_backbone.onnx` | 3,709,430 bytes（3.54 MiB） |
| Classification head ONNX（实验产物） | `onnx/mobilenet_v3_small_classification_head.onnx` | 6,464,110 bytes（6.16 MiB） |

完整模型为主要部署路径。Backbone/head 拆分用于结构和模块化实验，未用于生成上述完整 TensorRT engine。

## 验收状态与后续工作

- 完整 ONNX 文件已生成。
- 完整 TensorRT engine 文件已生成。
- 转换脚本记录的 engine 接口配置为固定 batch 1、FP32 输入张量 `[1, 3, 224, 224]`，输出 logits `[1, 1000]`。
- 当前工作区没有数值误差、任务精度、目标设备峰值内存、推理延迟或独立 TensorRT runtime 加载测试结果记录；这些属于 `03_model_validate` 的验证项目，不能由 engine 文件大小或生成成功推断。

Engine 是针对本机 Orin Nano、Jetson Linux 和 TensorRT 版本构建的目标运行时产物；更换设备或软件版本时应在目标环境重新构建并验证。
