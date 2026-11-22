# YOLO26n 目标检测模型流程

模型编号：`D-01`。当前交付为基础 `v1`：下载官方 COCO 预训练权重，导出并切分 ONNX，在 Jetson 上构建 FP16 TensorRT engine，再验证 PyTorch、ONNX 和 engine。当前不做项目数据微调、量化、CUDA Graph 或算子优化。

32 张 COCO 2017 `val2017` 图片只用于流程回归，不代表完整 COCO mAP。环境记录见 [`environment.md`](environment.md)。

## 按阶段复现

```bash
cd models/YOLO26n-Detection/01_model_train
python3 01-download_weights.py
python3 00-prepare_coco_val.py
python3 02-run_model.py
python3 03-evaluate_coco_val.py

cd ../02_model_convert
python3 02-check_layers.py
python3 01-export_onnx.py
python3 03-onnx_slice.py
python3 04-check_onnx.py
python3 05-build_tensorrt.py

cd ../03_model_validate
python3 01-numerical-validate.py
python3 02-task-validate.py
bash 03-memory-validation.sh
```

stage 02 会导出完整 ONNX 以及用于结构学习的 backbone+Neck/head 切分 ONNX；部署 engine 从完整 ONNX 构建。checkpoint、数据集和中间文件是本地产物，按 `.gitignore` 管理。

## 04 优化阶段

本版不运行性能 profiling、CUDA Graph、量化或算子优化。此阶段以后用于算子效率改进和自定义 TensorRT Plugin。

## 05 构建基础 C++ Demo

```bash
cd models/YOLO26n-Detection/05_model_demo/v1
mkdir -p models
cp ../../02_model_convert/engine/yolo26n_fp16.engine models/
cmake -S . -B build
cmake --build build -j2
./build/yolo26n_demo
```

Demo 对固定 `assets/sample.jpg` 做一次检测：OpenCV CPU 完成 letterbox，TensorRT 执行 engine，C++ 在 CPU 上进行置信度过滤和 NMS，然后打印检测框。它不依赖摄像头、用户界面或自定义 CUDA 核。engine 应在目标 Jetson/TensorRT 环境中构建。
