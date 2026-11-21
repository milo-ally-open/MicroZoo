# YOLO26n 目标检测部署流程

本目录按 MicroZoo 的五阶段流程实现 D-01：官方 YOLO26n COCO 预训练模型、ONNX、Jetson TensorRT FP16、数值与任务验证、目标硬件剖析，以及 TensorRT C++ 检测 demo。

当前使用官方预训练权重，**没有针对本项目数据微调**。32 张 COCO 2017 `val2017` 图片用于检查流程、转换回归和 demo，不是完整 COCO 验证集，不能据此宣称完整 COCO mAP。

## 环境

本次实测环境见 [environment.md](environment.md)。代码使用系统 Python 和 JetPack 提供的 PyTorch / CUDA / TensorRT，不要用 pip 替换 Jetson 系统运行时。

## 从仓库根目录复现

```bash
cd models/YOLO26n-Detection/01_model_train
python3 01-download_weights.py
python3 00-prepare_coco_val.py
python3 02-run_model.py
python3 03-evaluate_coco_val.py

cd ../02_model_convert
python3 02-check_layers.py
python3 01-export_onnx.py
python3 03-check_onnx.py
python3 04-build_tensorrt.py

cd ../03_model_validate
python3 01-numerical-validate.py
bash 03-memory-validation.sh

cd ../04_model_optimize/baseline
python3 01-compare.py
python3 02-bench.py
bash 03-nsight.sh
python3 04-repeat_bench.py

cd ../v1
python3 01-bench-cudagraph.py

cd ../../05_model_demo/baseline
python3 00-prepare_assets.py
cmake -S . -B build
cmake --build build -j2
./build/yolo26n_demo
```

模型权重与样本数据由单独脚本显式准备；推理、转换和验证脚本都只读本地文件，不会在运行中暗中下载。COCO subset 准备器从 COCO 官方 2017 val 标注中使用固定随机种子 `20261121` 选出 32 张图片，生成 YOLO 格式边界框、类别映射、`data.yaml` 和 `selection.txt`。完整原始标注包在准备时临时下载并删除；最终目录只保留选中的小型数据子集。

每阶段的脚本、指标和报告位于对应目录。TensorRT engine 与 checkpoint 是目标设备生成的本地二进制文件，按仓库规则忽略；demo 的运行目录要求将 stage 02 生成的 engine 复制到 `05_model_demo/baseline/models/`。

阶段 04 的数据解读见 [baseline analysis](04_model_optimize/baseline/baseline_analysis.md)、[optimization report](04_model_optimize/optimization_report.md) 和 [V1 CUDA Graph report](04_model_optimize/v1/optimization_report.md)。阶段 05 的摄像头构建与实测见 [demo report](05_model_demo/demo_report.md)。
