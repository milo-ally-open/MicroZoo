# YOLO26n — 阶段 03 验证报告

## 验证条件

- 同一目标设备：Jetson Orin Nano Super，PyTorch CUDA、ONNX Runtime 和 TensorRT。
- 固定输入：COCO `val2017` 子集首张图片 `sample.jpg`；Ultralytics letterbox 640、RGB、NCHW、`[0,1]`。
- 任务指标：同一固定 32 张 val 子集、640×640、batch 1；PyTorch FP32、ONNX Runtime CPU、TensorRT FP16。
- 数据范围限制：32 张图片和 242 个框仅用于 pipeline/regression smoke test，不代表完整 COCO val2017 的正式 benchmark。

## 数值一致性

`metrics/numerical.json` 对 ONNX CPU 和 TensorRT FP16 比较同一张图片上置信度不低于 0.001 的输出候选，并用相同预处理运行 PyTorch/ONNX/TensorRT 高层检测接口后，按同类、IoU≥0.5 匹配输出框。单图匹配结果：PyTorch 对 ONNX、TensorRT 均找到同类框；与 TensorRT 匹配框 IoU 约 `0.989`，最大坐标误差约 `2.58 px`，置信度差约 `0.043`。ONNX CPU 与 TensorRT FP16 原始有效候选的类别一致，整行 MAE 约 `0.156`，最大绝对误差约 `0.439`（主要来自框坐标），余弦相似度约 `0.9999999`。框坐标与置信度字段误差见 JSON。

该数值比较只有一个高置信目标，不能覆盖边界案例；后续应结合 32 张任务指标判断整体转换影响。

## 任务精度、内存与运行时

同一固定 32 张子集、batch 1、640×640 的实测结果：

| 后端 | Precision | Recall | mAP50 | mAP50-95 | 每图 inference | CUDA 峰值 allocated / reserved |
|---|---:|---:|---:|---:|---:|---:|
| PyTorch FP32 | 0.617 | 0.620 | 0.638 | 0.484 | 65.3 ms | 64.7 / 88.0 MiB |
| ONNX Runtime CPU | 0.651 | 0.562 | 0.631 | 0.491 | 105.9 ms | 不适用 |
| TensorRT FP16 | 0.660 | 0.569 | 0.627 | 0.494 | 16.1 ms | 56.5 / 88.0 MiB |

相对 PyTorch，TensorRT 在本小样本上 mAP50 低 `0.0111`、mAP50-95 高 `0.0104`；这类几十张图片的波动不能证明总体精度升降。数值单图结果也显示坐标高度接近但置信度有差异。部署前仍应在完整 COCO val2017 或项目独立数据上复测，不能将本表作为完整基准或精度验收结论。

TensorRT 10.3.0 成功反序列化并执行 FP16 engine；ONNX Runtime `1.23.0` 的 CPUExecutionProvider 成功执行 ONNX。Engine I/O 为 FP32 输入 `[1,3,640,640]` 与输出 `[1,300,6]`，内部允许 TensorRT 使用 FP16 kernel。兼容信息记录于 `metrics/numerical.json`。

`tegrastats` 以 1 秒间隔覆盖这次三后端验证，总 RAM 用量峰值约 `5519 MiB / 7620 MiB`，结束时 `5400 MiB`；GPU 频率样本约 `15–43%`，温度约 `48–51°C`，整机输入功耗约 `6.6–7.9 W`。这是验证期间系统级共享内存用量，不是单独的 TensorRT 进程内存；日志在 `metrics/memory/tegrastats.log`。

## 验收判断

三种运行时均能实际执行；数值和任务指标必须分开看。当前数据集是 32 张小样本，因此转换链路通过 smoke/regression 检查，但完整 COCO 精度要求仍未验收。若 ONNX/TensorRT 与 PyTorch 的小样本 mAP 差异明显，应先排查 end-to-end 检测头与预处理/后处理一致性，再决定是否用于应用部署。
