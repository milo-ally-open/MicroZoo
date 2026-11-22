# OWLv2 v1 — 模型分析

## 任务范围

本版本部署 Google 的 `google/owlv2-base-patch16-ensemble`，研究开放词汇、文本条件目标检测。输入是一张图像和一组文本查询，输出每个图像位置对应的候选框、分数和文本查询索引。不进行训练或微调。

## 模型结构与接口

- 模型卡标记约 0.2B 参数，视觉与文本编码使用 CLIP 系列架构；视觉 Transformer 的每个输出 token 都接检测框预测和分类特征。开放词汇类别分数由文本查询特征参与计算。[Google 模型卡](https://huggingface.co/google/owlv2-base-patch16-ensemble)
- 推荐权重：`model.safetensors`，模型仓库快照固定为 `cfd3195ba4ea9592eec887ded089f4c08eff231d`。许可标记为 Apache-2.0。[模型文件与许可](https://huggingface.co/google/owlv2-base-patch16-ensemble/tree/main)
- 官方图像预处理尺寸为 960×960，先缩放、归一化并补边；归一化均值为 `[0.48145466, 0.4578275, 0.40821073]`，标准差为 `[0.26862954, 0.26130258, 0.27577711]`。[预处理配置](https://huggingface.co/google/owlv2-base-patch16-ensemble/blob/main/preprocessor_config.json)
- ViT-B/16 的 960×960 输入产生 60×60 个视觉 patch，即 3600 个候选位置。模型文档给出的原始框张量为 `(batch, num_patches, 4)`；类别分数维度随文本查询数变化。[OWLv2 Transformers 文档](https://huggingface.co/docs/transformers/model_doc/owlv2)
- 官方文本检测后处理默认分数阈值为 0.1，按目标图像尺寸还原坐标；普通 `post_process_object_detection` 接口没有 NMS 阈值参数。分析脚本先保留官方后处理结果，后续依据重复框情况决定是否单独加入 NMS。[后处理接口](https://huggingface.co/docs/transformers/model_doc/owlv2)

## Jetson 环境与初步可行性

本机是 Jetson Orin Nano Super，系统 Python 3.10.12 中已有 NVIDIA PyTorch `2.5.0a0+872d972e41.nv24.08`，CUDA 12.6 且 `torch.cuda.is_available()` 为真；TensorRT 10.7.0。Conda `base` 使用 Python 3.14，不能直接复用 Python 3.10 的 Jetson PyTorch wheel。新建专用 Conda Python 3.10 环境，并通过 `.pth` 仅引用 Jetson 系统包目录；环境版本和创建命令见 [`environment.md`](environment.md)，不安装或替换 Torch。

**初步判断：**可以在设备上继续验证；是否适合实时摄像头取决于 960 输入下的实际峰值内存和时延。模型参数量和 patch 数量都明显高于已有 YOLO26n 基线，当前不预设实时 FPS。第一轮以 FP16 PyTorch 推理检查资源可行性；之后再单独量化和比较精度。

## 本轮运行记录

- 权重下载自固定仓库 revision `cfd3195ba4ea9592eec887ded089f4c08eff231d`，保存在本地 `checkpoints/`，不纳入 Git。
- 使用本地 COCO 2017 图片 `coco_000000039769.jpg`，尺寸 640×480；四个固定英文查询：cat、dog、person、remote control。
- PyTorch FP16 输入张量：`pixel_values=[1,3,960,960]`、文本 `input_ids=[4,16]`、`attention_mask=[4,16]`。
- 单次前向耗时约 **3020 ms**；图像处理器预处理约 **238 ms**；PyTorch 记录的峰值分配约 **942 MiB**。这是一次 smoke check，未预热、未重复统计，也未包含读图、主机到设备传输和后处理，不作为正式性能基准。
- 分数阈值 0.1 时得到 5 个候选框：两只猫分别约 0.663/0.612，遥控器候选约 0.798/0.691/0.144。预览和完整记录在 [`outputs/first-inference.jpg`](outputs/first-inference.jpg) 与 [`outputs/first-inference.json`](outputs/first-inference.json)。
- 该图的检测类别和位置符合预期，说明 Conda + Jetson CUDA PyTorch + OWLv2 的基础推理链路可运行；单图约 3 秒也显示默认 960 输入下实时性有明显风险。此结果仅供可行性判断，不能代替稳态 profiling 或任务精度评估。

## 本阶段计划与验收

1. 固定模型仓库 revision、处理器配置、查询词和示例图片。
2. 检查系统 CUDA PyTorch 与隔离 Transformers 环境，下载固定 revision 到本地忽略目录。**已完成。**
3. 运行一张 COCO 图片的文本查询推理并检查设备内存。**已完成 smoke check；尚未形成精度或稳态性能指标。**
4. 准备小型、带框标注且查询词固定的开放词汇测试集，评估任务效果并确定输入尺寸与运行精度。

本阶段完成条件：在 Jetson GPU 上成功加载固定 revision；记录实际输入输出形状、检测结果、内存和延迟；用固定查询词与有标注测试集形成任务基线。ONNX/TensorRT 转换兼容性留到 `02_model_convert` 验证。
