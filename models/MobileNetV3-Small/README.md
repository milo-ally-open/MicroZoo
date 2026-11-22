# MobileNetV3-Small 模型研发流程

**编号：** `C-01`（分类模型） · **任务：** ImageNet-1K 单标签图像分类 · **基线：** Torchvision `MobileNet_V3_Small_Weights.IMAGENET1K_V1` 预训练权重

本项目只下载并部署预训练模型，不进行训练或微调。输入按 Torchvision 预处理规则缩放、中心裁剪并归一化为 `[1, 3, 224, 224]`，模型输出 1000 类 logits。研发步骤和环境记录分别见下文及 [`environment.md`](environment.md)。

## 目录与步骤

| 步骤 | 目录 | 内容 | 记录 |
|---|---|---|---|
| 01 模型获取与单次运行 | [`01_model_train/`](01_model_train/) | 下载 ImageNet-1K 预训练权重；用 `dog.jpg` 检查 CUDA 加载、推理和 Top-5 输出。目录名沿用流程编号，本步骤不训练模型。 | — |
| 02 模型转换 | [`02_model_convert/`](02_model_convert/) | 导出完整 ONNX、检查中间层、拆分 backbone/分类头，并提供 TensorRT FP16 构建脚本。 | [`conversion_report.md`](02_model_convert/conversion_report.md) |
| 03 模型验证 | [`03_model_validate/`](03_model_validate/) | 对比 ONNX Runtime CPU 与 TensorRT FP16 logits，并检查单张样例的分类结果。 | [`validation_report.md`](03_model_validate/validation_report.md) |
| 04 模型优化 | [`04_model_optimize/`](04_model_optimize/) | 用 TensorRT baseline 做性能剖析；V1 对固定形状推理尝试 CUDA Graph。 | [`baseline_analysis.md`](04_model_optimize/baseline/baseline_analysis.md)、[`optimization_report.md`](04_model_optimize/v1/optimization_report.md) |
| 05 端侧 Demo | [`05_model_demo/`](05_model_demo/) | 在 Jetson 上用 C++、CUDA、OpenCV 和 TensorRT 运行 baseline 与 CUDA Graph V1。 | [`baseline README`](05_model_demo/baseline/README.md)、[`V1 README`](05_model_demo/v1/README.md) |

## 快速复现

以下命令在 Linux、WSL 或 Jetson 的 bash 中执行。Torch 推理步骤需要可用的 CUDA GPU；TensorRT engine 必须在实际部署设备的软件栈上构建。

### 01 下载权重并运行样例

```bash
cd models/MobileNetV3-Small/01_model_train
python 01-download_weights.py
python 02-run_model.py
```

下载脚本将权重放在 `01_model_train/checkpoints/`。单张 `dog.jpg` 的推理仅确认权重加载和分类流程，不代表独立测试集上的准确率或 F1。

### 02 导出 ONNX 与拆分模型

导出脚本从 `02_model_convert/pt/` 读取权重，因此先复制下载的 checkpoint：

```bash
cd models/MobileNetV3-Small/02_model_convert
mkdir -p pt
cp ../01_model_train/checkpoints/mobilenet_v3_small-047dcff4.pth pt/mobilenet_v3_small-047dcff4.pt
python 01-export_onnx.py
python 02-check_layers.py
python 03-onnx_slice.py
```

完整模型 ONNX 固定输入为 `[1, 3, 224, 224]`；切分脚本另生成 backbone 和分类头 ONNX。当前 `04-build_tensorrt.py` 读取的是 `mobilenet_v3_small_backbone.onnx`，输出 `mobilenet_v3_small_backbone.engine`，所以它只构建拆分后的 backbone。

**完整模型 engine 的复现入口目前没有与报告对应。** `conversion_report.md` 记录了完整 FP16 engine，但当前转换目录中的 TensorRT 脚本并不生成该文件。仓库中的 C++ demo 带有已构建的完整模型 engine；TensorRT engine 与设备、TensorRT/CUDA 版本相关，迁移到其他 Jetson 时需要在目标环境重新构建并验证。

### 03 数值与任务验证

验证脚本需要完整 ONNX 和完整模型 engine 位于本步骤的 `onnx/` 与 `engine/` 子目录。它们不会自动从前序步骤复制文件：

```bash
cd models/MobileNetV3-Small/03_model_validate
mkdir -p onnx engine
cp ../02_model_convert/onnx/mobilenet_v3_small.onnx onnx/
cp ../05_model_demo/baseline/models/mobilenet_v3_small_fp16.engine engine/
python 01-numerical-validate.py
python 02-task-validate.py
```

当前记录的 `dog.jpg` 上，ONNX 与 TensorRT FP16 的 Top-1 均为 Samoyed，logits 余弦相似度为 `0.99994946`。该单图结果是功能检查，不是数据集精度验收。

### 04 性能剖析与优化

baseline 脚本从当前目录的 `weights/` 和 `engine/` 读取权重与 engine：

```bash
cd models/MobileNetV3-Small/04_model_optimize/baseline
mkdir -p weights engine
cp ../../01_model_train/checkpoints/mobilenet_v3_small-047dcff4.pth weights/
cp ../../05_model_demo/baseline/models/mobilenet_v3_small_fp16.engine engine/
python 01-compare.py
python 02-bench.py
```

`03-nsight.sh` 用于可选的 Nsight Systems 采集，需先检查脚本中的命令及本机 Nsight 安装。CUDA Graph V1 使用 baseline engine 和样例图：

```bash
cd models/MobileNetV3-Small/04_model_optimize/v1
python 01-bench-cudagraph.py
```

已记录的 V1 Jetson 实测中，CUDA Graph 三轮 median 的中位数比 eager 低约 `18.6%`；测试为固定输入下的模型提交耗时，不是包含解码、预处理及传输的端到端 SLA。设备 SKU 未在 baseline profile 记录中注明，详见两份优化报告。

### 05 在 Jetson 上构建 C++ Demo

在目标 Jetson 上分别进入 `05_model_demo/baseline/` 或 `05_model_demo/v1/`，按各自 README 编译运行。两份目录包含完整模型 TensorRT engine、`dog.jpg` 和 ImageNet-1K 类别文件；engine 是目标运行时产物，不应假定可跨设备复用。

## 验收范围

- 权重可加载，模型可以对固定样例完成一次 CUDA 推理并输出 Top-5。
- 完整 ONNX 和 TensorRT FP16 的数值差异及单样例分类结果见验证报告。
- baseline 与 CUDA Graph 性能记录见优化报告；样本量、设备状态和计时范围限制以报告为准。
- 单张样例不能用于宣称分类准确率、召回率或 F1。正式任务精度需要独立标注测试集和预先约定的指标门槛。
