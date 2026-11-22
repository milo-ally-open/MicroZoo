# YOLO26n-cls 项目

本目录按 MicroZoo 五阶段流程记录 YOLO26n-cls 从官方预训练权重到 Jetson TensorRT C++ demo 的可复现过程。模型用于 ImageNet-1K 单标签分类；本项目没有使用 Imagenette 子集微调模型。

## 目录

- `01_model_train/`：下载权重、准备固定 100 张 ImageNette 子集、PyTorch 推理和精度/内存记录。
- `02_model_convert/`：导出完整 ONNX、检查层、切分 backbone/head、构建 Jetson TensorRT FP16 engine。
- `03_model_validate/`：ONNX Runtime CPU 与 TensorRT 数值比较、100 张任务精度比较、目标设备内存采样。
- `04_model_optimize/`：TensorBoard、TensorRT 逐层 profile、Nsight Systems、重复基线测量和 CUDA Graph V1。
- `05_model_demo/`：TensorRT C++ baseline 与 CUDA Graph V1，均含 CUDA/OpenCV 两种图像预处理路径。
- `environment.md`：记录目标设备和软件版本。

## 首次准备

在 Jetson 的系统 Python 环境运行。JetPack 已提供的 PyTorch、CUDA、TensorRT 和 OpenCV 不要被 pip 替换；只将 Ultralytics 固定到项目使用的版本：

```bash
python3 -m pip install --no-deps -r models/YOLO26n-CLS/requirements.txt
```

权重和 Imagenette 小型子集需要显式下载。测试集脚本固定使用每个 ImageNet synset 目录中按文件名排序的前 10 张图，共 100 张；清单写入 `01_model_train/datasets/imagenette2-small/selection.txt`。数据目录被 Git 忽略，可随时重新生成。`sample.jpg` 是 tench 类测试图，数据准备脚本会复制到各阶段的工作目录。

```bash
cd models/YOLO26n-CLS/01_model_train
python3 01-download_weights.py
bash 00-download_imagenette.sh
python3 02-run_model.py
python3 03-evaluate_imagenette.py
```

## 按阶段执行

```bash
cd models/YOLO26n-CLS/02_model_convert
python3 01-export_onnx.py
python3 02-check_layers.py
python3 03-onnx_slice.py
python3 04-build_tensorrt.py

cd ../03_model_validate
python3 01-numerical-validate.py
python3 02-task-validate.py
bash 03-memory-validation.sh

cd ../04_model_optimize/baseline
python3 01-compare.py
python3 02-bench.py
bash 03-nsight.sh
python3 04-repeat_bench.py
tensorboard --logdir=profiling

cd ../v1
python3 01-bench-cudagraph.py
```

完整流程及已测数据见各阶段的 `*_report.md`。转换后的 ONNX、checkpoint、validation 副本和 profiling 原始文件按仓库 `.gitignore` 规则管理；模型下载和转换脚本可重建它们。TensorRT engine 与构建设备/软件栈绑定。

## C++ demo

```bash
cd models/YOLO26n-CLS/05_model_demo/baseline
cmake -S . -B build
cmake --build build -j2
./build/yolo26n_cls_demo

cd ../v1
cmake -S . -B build
cmake --build build -j2
./build/yolo26n_cls_v1_demo

bash 03-system-validation.sh
python3 04-make-demo-video.py
```

两个 demo 都需要在各自目录运行，以相对路径读取 engine、`assets/sample.jpg` 和 ImageNet 类别表。完整构建说明见 `05_model_demo/baseline/README.md` 与 `05_model_demo/v1/README.md`。

## 关键结果

- 固定 100 张 ImageNette 子集：PyTorch、ONNX Runtime CPU、TensorRT FP16 的 Top-1 均为 64%，Top-5 均为 82%。这只是 10 类小子集回归结果，不代表完整 ImageNet-1K 精度。
- ONNX Runtime CPU 与 TensorRT FP16 在固定图像上的 MAE 为 `8.31e-7`，最大绝对误差为 `3.79e-4`，Top-1 一致。
- backbone 输出 `[1,256,7,7]`，classification head 输出 `[1,1000]`；两段 ONNX 串接与完整 ONNX 的 Top-1 一致。
- CUDA Graph 在 trtexec 重复测量中的 latency median 中位数比 eager 低约 32.4%；测量 CV 较高，结果仅用于优化方向判断，详见 V1 报告。
