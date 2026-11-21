# YOLO26n-cls VALIDATION

## 固定条件

- 原始 checkpoint：`01_model_train/checkpoints/yolo26n-cls.pt`；同一文件复制到 `02_model_convert/pt/`。
- 输入：RGB FP32 NCHW，`[1, 3, 224, 224]`，Ultralytics `classify_transforms(224)`；像素在 `[0,1]`，不做 ImageNet mean/std 归一化。
- ONNX Runtime：CPUExecutionProvider；TensorRT：Jetson 本机 FP16 engine。
- 数值验证：本地 `sample.jpg`；对比 ONNX Runtime CPU 与 TensorRT FP16 输出，并比较 full ONNX 与 backbone/head 两段 ONNX 串接输出。
- 任务验证：固定 ImageNette2-160 `val` split 每类按文件名排序的前 10 张，共 100 张、10 个 ImageNet synset；类别目录映射到 ImageNet-1K 原始类别索引，对比 PyTorch 参考 JSON、ONNX Runtime CPU、TensorRT FP16 的 Top-1 / Top-5。
- 不把 10 类子集的准确率解释为完整 ImageNet-1K 精度。

## 数值一致性

固定 `sample.jpg` 上 ONNX Runtime CPU vs TensorRT FP16：MAE `8.31e-7`，最大绝对误差 `3.79e-4`，平均相对误差 `0.225`，余弦相似度 `0.99999994`，Top-1 一致。平均相对误差受接近 0 的类别概率影响，应结合绝对误差、余弦相似度和 Top-1 一致性判断。

完整 ONNX vs backbone/head 两段 ONNX 串接：MAE `4.36e-10`，最大绝对误差 `2.38e-7`，Top-1 一致。

## 任务精度

固定 100 张 ImageNette 子集的任务结果：

| Backend | Top-1 | Top-5 | 相对 PyTorch |
|---|---:|---:|---:|
| PyTorch CUDA FP32 | 64% | 82% | reference |
| ONNX Runtime CPU | 64% | 82% | 0.0 pp / 0.0 pp |
| TensorRT FP16 | 64% | 82% | 0.0 pp / 0.0 pp |

`datasets/imagenette2-small/selection.txt` 记录选中的图片；类别映射、图片排序和预处理均固定。该 100 张子集只用于转换任务精度回归，不能代表完整 ImageNet-1K 精度。

## 目标设备内存与兼容性

Engine 由 TensorRT 10.3.0 在目标 Jetson 上构建，完整 ONNX 和 engine 均已在 ONNX Runtime CPU / TensorRT 运行时加载并推理，没有 OOM。Engine I/O：`images` FP32 `[1,3,224,224]` → `output0` FP32 `[1,1000]`。最近一次完整验证的 `tegrastats` 记录共 248 个采样点，数值与 100 张任务验证期间总 RAM 峰值约 4,871 / 7,620 MiB；该数值包含 OS、ONNX Runtime 和 Python 进程，不是 engine 单独占用，也不代表空闲设备基线。阶段 01 的 PyTorch batch-32 峰值 CUDA 分配为 159.3 MiB。

## 复现命令

```bash
cd models/YOLO26n-CLS/03_model_validate
python3 01-numerical-validate.py
python3 02-task-validate.py
bash 03-memory-validation.sh
```
