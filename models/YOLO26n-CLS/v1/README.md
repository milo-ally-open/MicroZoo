# YOLO26n-cls 分类模型流程

模型编号：`C-01`。当前交付为基础 `v1`：下载官方 ImageNet-1K 预训练权重，导出和切分 ONNX，在 Jetson 上构建 FP16 TensorRT engine，并验证 PyTorch、ONNX 和 TensorRT 推理。项目不做模型训练或微调；不做量化、CUDA Graph 或算子优化。

## 阶段目录

| 阶段 | 目录 | 内容 |
|---|---|---|
| 01 权重获取 | [`01_model_train/`](01_model_train/) | 下载权重、准备固定 ImageNette 样例集，运行 PyTorch 检查。此版本不训练。 |
| 02 模型转换 | [`02_model_convert/`](02_model_convert/) | 导出完整 ONNX、检查层、切分 backbone/head、构建完整 TensorRT FP16 engine。 |
| 03 模型验证 | [`03_model_validate/`](03_model_validate/) | 验证 ONNX 与 engine 数值和任务结果，并记录目标设备内存。 |
| 04 模型优化 | [`04_model_optimize/`](04_model_optimize/) | 当前暂缓；未来用于算子效率优化与 TensorRT Plugin。 |
| 05 端侧 Demo | [`05_model_demo/`](05_model_demo/) | 单张样例、OpenCV CPU 预处理和 TensorRT 分类的基础 C++ 示例。 |

环境和依赖见 [`environment.md`](environment.md) 与 [`requirements.txt`](requirements.txt)。数据、checkpoint、ONNX 和 engine 都由脚本生成并按 `.gitignore` 保存在本地；仓库只保留复现脚本、Demo 源码和固定样例。

## 按阶段复现

在 Jetson 的系统 Python 环境执行。JetPack 提供的 PyTorch、CUDA、TensorRT 和 OpenCV 不要用 pip 替换；仅安装项目固定的 Ultralytics 版本：

```bash
python3 -m pip install --no-deps -r models/YOLO26n-CLS/v1/requirements.txt

cd models/YOLO26n-CLS/v1/01_model_train
python3 01-download_weights.py
bash 00-download_imagenette.sh
python3 02-run_model.py
python3 03-evaluate_imagenette.py

cd ../02_model_convert
python3 01-export_onnx.py
python3 02-check_layers.py
python3 03-onnx_slice.py
python3 04-build_tensorrt.py

cd ../03_model_validate
python3 01-numerical-validate.py
python3 02-task-validate.py
bash 03-memory-validation.sh
```

PyTorch 与转换后后端的检查结果见各阶段报告。ImageNette 固定子集用于流程回归，不代表完整 ImageNet-1K 精度。

## 04 优化阶段

本版不运行 profiling、CUDA Graph、量化或算子优化。此阶段以后留给算子效率改进和自定义 TensorRT Plugin。

## 05 构建基础 C++ Demo

```bash
cd models/YOLO26n-CLS/v1/05_model_demo
mkdir -p models
cp ../02_model_convert/engine/yolo26n-cls_fp16.engine models/
cmake -S . -B build
cmake --build build -j2
./build/yolo26n_cls_demo
```

Demo 读取固定 `assets/sample.jpg`，用 OpenCV 在 CPU 预处理，运行一次 TensorRT 推理并输出分类结果。没有 CUDA 预处理分支、性能比较或 CUDA Graph。engine 与目标 Jetson/TensorRT 版本绑定，迁移时从 ONNX 重新构建。
