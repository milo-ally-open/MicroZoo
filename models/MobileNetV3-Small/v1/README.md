# MobileNetV3-Small 模型流程

编号：`C-01`（分类模型）。当前交付为基础 `v1`：下载 Torchvision ImageNet-1K 预训练权重，导出并切分 ONNX，在目标 Jetson 上构建 FP16 TensorRT engine，再分别检查 PyTorch 权重、ONNX 和 engine。当前不训练、不量化、不使用 CUDA Graph，也不做算子优化。

## 阶段目录

| 阶段 | 目录 | 内容 |
|---|---|---|
| 01 权重获取 | [`01_model_train/`](01_model_train/) | 下载预训练权重，用固定样例确认 PyTorch 模型能加载和推理；本项目不训练。 |
| 02 模型转换 | [`02_model_convert/`](02_model_convert/) | 导出完整 ONNX、检查层、切分 backbone/head，并由完整 ONNX 构建一个 FP16 engine。 |
| 03 模型验证 | [`03_model_validate/`](03_model_validate/) | 检查 ONNX 与 TensorRT engine 数值和样例分类；PyTorch checkpoint 的运行检查在阶段 01。 |
| 04 模型优化 | [`04_model_optimize/`](04_model_optimize/) | 当前暂缓；未来用于算子效率优化与 TensorRT Plugin。 |
| 05 端侧 Demo | [`05_model_demo/`](05_model_demo/) | 单张图片、OpenCV CPU 预处理、TensorRT 分类的基础 C++ 示例。 |

环境版本见 [`environment.md`](environment.md)。下载权重、ONNX、engine 等生成文件按 `.gitignore` 规则保存在本地；仓库只保留复现脚本、Demo 源码和固定样例。

## 01 下载并运行 PyTorch 权重

```bash
cd models/MobileNetV3-Small/v1/01_model_train
python 01-download_weights.py
python 02-run_model.py
```

`02-run_model.py` 使用 `dog.jpg` 做一次 PyTorch/CUDA 推理。单张图片是加载与执行检查，不代表数据集精度。

## 02 导出、切分并构建 TensorRT engine

```bash
cd models/MobileNetV3-Small/v1/02_model_convert
mkdir -p pt
cp ../01_model_train/checkpoints/mobilenet_v3_small-047dcff4.pth pt/mobilenet_v3_small-047dcff4.pt
python 01-export_onnx.py
python 02-check_layers.py
python 03-onnx_slice.py
python 04-build_tensorrt.py
```

`01-export_onnx.py` 生成完整分类模型 ONNX；`03-onnx_slice.py` 另生成 backbone 和 classification head 两个 ONNX，用于学习和检查切分边界。`04-build_tensorrt.py` 从**完整模型 ONNX**生成一个 engine；Demo 使用此 engine，不将两个切分图串联部署。engine 应在目标 Jetson 的 TensorRT 环境中构建。

## 03 验证 ONNX 与 engine

```bash
cd models/MobileNetV3-Small/v1/03_model_validate
mkdir -p onnx engine
cp ../02_model_convert/onnx/mobilenet_v3_small.onnx onnx/
cp ../02_model_convert/engine/mobilenet_v3_small_fp16.engine engine/
python 01-numerical-validate.py
python 02-task-validate.py
```

验证使用本地 `dog.jpg`。当前报告是单样例流程检查，不替代独立标注测试集上的分类精度评估。

## 04 优化阶段

本版不在 `04_model_optimize` 运行 profiling、CUDA Graph、量化或其他优化。后续真正开展算子效率改进或 TensorRT Plugin 时，再在此阶段建立独立记录。

## 05 构建基础 C++ Demo

```bash
cd models/MobileNetV3-Small/v1/05_model_demo
mkdir -p models
cp ../02_model_convert/engine/mobilenet_v3_small_fp16.engine models/
cmake -S . -B build
cmake --build build -j2
./build/mobilenet_demo
```

Demo 从当前目录读取 `models/mobilenet_v3_small_fp16.engine`、`assets/dog.jpg` 和 ImageNet 类别表，执行一次分类并输出 Top-5。更换 Jetson 或 TensorRT 版本时，应重新构建 engine。
