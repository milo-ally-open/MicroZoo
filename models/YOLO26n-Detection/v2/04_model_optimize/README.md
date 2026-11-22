# 阶段 04 — TensorRT INT8 性能剖析

本阶段在目标 Jetson 上用同一张留出图片分别采集原始 PyTorch FP32 和 v2 TensorRT INT8 的 profiler trace。TensorBoard worker 名分别为 `PyTorch_FP32` 和 `TensorRT_INT8`。此外，TensorRT `IProfiler` 输出融合层耗时，便于定位 engine 内的慢层。

```bash
cd models/YOLO26n-Detection/v2/04_model_optimize
sudo -E python3 01-profile.py
tensorboard --logdir=profiling
```

脚本先 warmup，再生成 TensorBoard trace 和 `profiling/tensorrt_int8_layers.json`。PyTorch profiler CUDA 活动依赖 Jetson 上的 CUPTI 权限与 PyTorch 构建配置。此处为剖析，不作为稳定延迟 benchmark；正式性能结论应在相同功耗/时钟条件下重复测量。

剪枝和 INT8 量化是本 v2 的模型级改动；本阶段不再额外改算子或 CUDA kernel。瓶颈结论应结合 TensorBoard trace、TensorRT layer profile 和阶段 03 的精度结果记录在 `optimization_report.md`。
