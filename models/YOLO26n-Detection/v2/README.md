# YOLO26n Detection v2

本版从 v1 的官方 COCO 预训练模型出发，不进行训练或微调。流程是：导出完整 FP32 ONNX → ONNX 结构化通道裁剪 → TensorRT EntropyCalibrator2 PTQ → INT8 engine → 数值/任务/内存验证 → profiler 分析 → 摄像头实时 C++ Demo。

剪枝脚本只改一个已在 Netron 中检查过的中间特征：`/model.7/conv/Conv` 的 256 个输出通道按滤波器 L1 范数保留 250 个（裁剪约 2.3%），并同步裁剪后续 `/model.8/cv1/conv/Conv` 的输入权重。脚本会验证 Conv → Sigmoid/Mul → Conv 的连接关系；模型图改变时会停止，不会猜测依赖关系。这是结构化通道裁剪，不是把权重置零的稀疏化。该保守剪枝配置将 FP32 ONNX 的留出集 mAP50 从 0.802 降到 0.764，但 ONNX 文件仅缩小约 0.04%，因此剪枝的体积收益很小。

不做微调意味着剪枝可能损失检测精度。INT8 校准只使用 24 张图片，另外 8 张留作独立验证；这是流程试跑用的小样本，不足以代表正式精度。实际验证中 TensorRT INT8 的 mAP50 为 0.682，PyTorch FP32 为 0.765，出现约 0.083 的下降。本 v2 已完成 PTQ 构建、验证、profiler 和摄像头 demo 的编译，但当前小样本精度不能判定为达标。正式实验应增加覆盖实际场景的校准图片，并由 `03_model_validate` 评估任务精度。

## 准备数据与权重

先准备 v1 权重和 COCO 小数据集，再把 checkpoint 放入 v2（这里只复制 PT 文件，不复制 ONNX）：

```bash
cd models/YOLO26n-Detection/v1/01_model_train
python3 01-download_weights.py
python3 00-prepare_coco_val.py

mkdir -p ../../v2/01_model_train/checkpoints ../../v2/02_model_convert/pt
cp checkpoints/yolo26n.pt ../../v2/01_model_train/checkpoints/
cp checkpoints/yolo26n.pt ../../v2/02_model_convert/pt/
```

## 复现五阶段

准备 v2 的校准集与独立留出集，然后按顺序导出 FP32 ONNX、剪枝、准备校准输入并构建 INT8 engine：

```bash
cd ../../v2/01_model_train
python3 01-prepare-calibration.py

cd ../02_model_convert
python3 01_export_onnx.py
python3 02-prune-onnx.py
python3 03-prepare-calibration.py
python3 04-build-tensorrt-int8.py

cd ../03_model_validate
python3 01-numerical-validate.py
python3 02-task-validate.py
bash 03-memory-validation.sh

cd ../04_model_optimize
sudo -E python3 01-profile.py
tensorboard --logdir=profiling

cd ../05_model_demo
mkdir -p models
cp ../02_model_convert/engine/yolo26n_pruned_int8.engine models/
cmake -S . -B build
cmake --build build -j2
./build/yolo26n_int8_demo
```

## 转换说明

`01_export_onnx.py` 从 `pt/yolo26n.pt` 导出完整 FP32 ONNX，固定输入为 batch 1、640×640，opset 17，不导出 NMS。脚本把文件统一保存为 `onnx/yolo26n_fp32.onnx`，并用 ONNX checker 检查图。

`02-prune-onnx.py` 读取这份完整 ONNX 并执行结构化通道裁剪。`03-prepare-calibration.py` 按实际推理预处理校准图像，生成 `[N,3,640,640]` 的 FP32 `[0,1]` 输入。

`04-build-tensorrt-int8.py` 直接使用 TensorRT 的 `IInt8EntropyCalibrator2`。校准器逐批把预处理好的 FP32 图片送入 TensorRT；TensorRT 在构建 engine 时运行校准前向、收集激活直方图并计算量化范围，不需要逐个暴露 ONNX 中间 tensor，也不需要 PPQ 或手工 MinMax JSON。TensorRT 文档推荐 EntropyCalibrator2 用于 CNN。PPQ 仓库也展示了代表性图片的 TRT_INT8 校准流程，但本项目直接调用 TensorRT calibrator，不安装或调用 PPQ。[PPQ YOLO PTQ 示例](https://github.com/OpenPPL/ppq/blob/master/ppq/samples/Yolo/02_Quantization.py)

当前 ONNX 是静态 batch 1，所以校准 batch 固定为 1。TensorRT 的校准表保存在 `calibration/yolo26n_int8_entropy.cache` 并在后续 build 中复用。若更换剪枝 ONNX、校准图片或预处理方式，先删除该缓存再构建，避免重用与当前模型不匹配的校准表。

注意：当前目标设备使用 TensorRT 10.7。原生 calibrator 属于 implicit INT8 API，TensorRT 10 已将该路径标记为 deprecated，TensorRT 11 已移除这些 calibrator API。这里使用原生接口是在当前 TRT 10.x 环境中让 builder 自己完成 PTQ；若升级到 TRT 11，应迁移到带 Q/DQ 的显式量化模型。见 [TensorRT 10 量化说明](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/work-quantized-types.html)、[10.3 弃用说明](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/release-notes-10/10.3.0.html) 和 [TensorRT 11 迁移说明](https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x-python-api-patterns.html)。

TensorRT build 使用 `build_serialized_network`；engine 需在目标 Jetson 上构建。ONNX、校准图像、calibration cache、engine、验证日志、profiler trace 和 C++ build 均由 ignore 规则排除。阶段 03 分别报告 FP32 剪枝误差与 INT8 数值误差，并用 8 张留出图做任务回归；阶段 04 提供 TensorRT layer timing 和 TensorBoard trace；阶段 05 沿用 v1 摄像头模式（YUYV 640×480@30），通过 GStreamer/V4L2 采集并用 VIC 转换格式，CUDA 预处理、CUDA NMS/画框、分阶段计时和实时 FPS，详见其 README。

24 张校准图和 8 张留出图只用于复现小样本流程，不能据此宣称正式 COCO 精度达标。所有转换和验证的版本、实测结果需补录到各阶段报告。
