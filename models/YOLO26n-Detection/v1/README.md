# YOLO26n 目标检测模型流程

模型编号：`D-01`。当前交付为基础 `v1`：下载官方 COCO 预训练权重，导出并切分 ONNX，在 Jetson 上构建 FP16 TensorRT engine，再验证 PyTorch、ONNX 和 engine。当前不做项目数据微调、量化、CUDA Graph 或算子优化。

32 张 COCO 2017 `val2017` 图片只用于流程回归，不代表完整 COCO mAP。环境记录见 [`environment.md`](environment.md)。

## 按阶段复现

```bash
cd models/YOLO26n-Detection/v1/01_model_train
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

## 05 构建实时 C++ Demo

```bash
cd models/YOLO26n-Detection/v1/05_model_demo
mkdir -p models
cp ../02_model_convert/engine/yolo26n_fp16.engine models/
cmake -S . -B build
cmake --build build -j2
./build/yolo26n_demo
```

Demo 使用设备 `0` 的摄像头做实时检测，在窗口中绘制检测框，并每秒在窗口和终端打印 FPS。OpenCV CPU 完成 letterbox 和 NCHW 转换，TensorRT 执行 FP16 engine；CUDA kernel 完成置信度过滤、坐标还原、两两 IoU 计算、greedy class-aware NMS 和矩形边框绘制。CPU 只叠加类别文字与 FPS。按 `q` 或 `Esc` 退出。

engine 必须在目标 Jetson/TensorRT 环境中构建。更换 TensorRT 版本后，先运行 `../02_model_convert/05-build_tensorrt.py` 重新构建 engine，再复制到 `05_model_demo/models/`。
