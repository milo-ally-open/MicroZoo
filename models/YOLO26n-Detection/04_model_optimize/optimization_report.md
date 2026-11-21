# YOLO26n — 阶段 04 优化报告

## 结论

目标硬件为 Jetson Orin Nano Super。TensorRT FP16 固定 shape engine 的 profiling 显示 eager 调用有较高的 host enqueue 开销；采用 CUDA Graph replay 后，三轮 `trtexec` 的 median latency 中位数从 `4.302 ms` 降至 `4.010 ms`，enqueue median 从 `3.270 ms` 降至 `0.021 ms`。Python 同步测量也观察到中位延迟下降。V1 不修改权重、计算图、输入精度或后处理，因此保持同一 engine 输出。

测量仍有约 `25–35%` 的 GPU latency CV 和显著 P95 长尾。CUDA Graph 解决的是重复提交开销，不是 GPU 计算时间；当前收益方向明确，但不能据单次测量声称延迟稳定或代表完整摄像头应用吞吐。

## 阶段交付物

- 基线 trace、TensorRT layer profile、Nsight Systems 报告和重复测量：[`baseline/README.md`](baseline/README.md)、[`baseline/baseline_analysis.md`](baseline/baseline_analysis.md)。
- CUDA Graph 实现、实测对照和数值一致性：[`v1/optimization_report.md`](v1/optimization_report.md)。
- 同一 TensorRT FP16 engine 的任务精度和设备内存基准：[`../03_model_validate/validation_report.md`](../03_model_validate/validation_report.md)。V1 在固定 32 张 COCO 子集上的 eager 与 Graph engine 原始输出逐元素相同，因此任务指标维持为 Precision `0.660`、Recall `0.569`、mAP50 `0.627`、mAP50-95 `0.494`；这只是小样本回归值，不是完整 COCO 精度验收。
- 实时摄像头部署演示：[`../05_model_demo/demo_report.md`](../05_model_demo/demo_report.md)。

阶段 03 TensorRT FP16 的 CUDA 峰值 allocated/reserved 为 `56.5/88.0 MiB`，验证期间系统 RAM 峰值约 `5519/7620 MiB`。V1 继续使用同一 engine 与固定输入输出 buffer，增加 CUDA Graph executable；没有单独隔离其进程级内存增量，因此不把阶段 03 的数据当作 Graph 的独立内存测量。

## 后续建议

先在目标应用的摄像头链路上复测 CUDA Graph 版本的端到端处理吞吐，并保留当前 baseline 作为对照。只有在统一功耗、温度和负载条件下确认还有明显瓶颈后，再考虑降低输入分辨率、调整流水线或开发 TensorRT plugin/CUDA kernel。
