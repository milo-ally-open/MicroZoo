# YOLO26n TensorRT 基线剖析

所有 trace、TensorRT layer profile、Nsight 报告和重复基准均写入本目录的 `profiling/`，方便 TensorBoard 与 Nsight 工具直接查看。

```bash
python3 01-compare.py
python3 02-bench.py
bash 03-nsight.sh
python3 04-repeat_bench.py
tensorboard --logdir=profiling
```

TensorBoard PyTorch Profiler 的 worker 名称分别为 `PyTorch_Model` 和 `TensorRT_Engine`。`02-bench.py` 的 LayerSummary 是 TensorRT engine 的 IProfiler 层时间；不能把它和 PyTorch 图表混淆，也不能把 profile 层时间简单相加当作端到端 latency。正式延迟来自不挂 profiler 的 `trtexec` 三轮记录。Nsight Systems 用 `nsys stats` 或 GUI 打开 `profiling/nsys/*.nsys-rep`。
