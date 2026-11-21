"""以真实预处理后的 sample.jpg 对 TensorRT engine 做 IProfiler 逐层分析。"""

import json
import shutil
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch
import torch.profiler
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from ultralytics.data.augment import classify_transforms


class LayerProfiler(trt.IProfiler):
    def __init__(self):
        trt.IProfiler.__init__(self)
        self.layer_times = {}

    def report_layer_time(self, layer_name, milliseconds):
        self.layer_times[layer_name] = float(milliseconds)


project_dir = Path(__file__).resolve().parent
model_dir = project_dir.parent.parent
source_engine = model_dir / "02_model_convert" / "engine" / "yolo26n-cls_fp16.engine"
engine_path = project_dir / "engine" / source_engine.name
image_path = project_dir / "sample.jpg"
profile_dir = project_dir / "profiling"
layer_log_dir = profile_dir / "TensorRT_Bench"
profile_dir.mkdir(parents=True, exist_ok=True)
if not engine_path.is_file():
    shutil.copy2(source_engine, engine_path)

input_tensor = classify_transforms(224)(Image.open(image_path).convert("RGB")).unsqueeze(0).cuda()
runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
if engine is None:
    raise RuntimeError("TensorRT 无法加载 engine")
context = engine.create_execution_context()
input_names = [
    engine.get_tensor_name(index)
    for index in range(engine.num_io_tensors)
    if engine.get_tensor_mode(engine.get_tensor_name(index)) == trt.TensorIOMode.INPUT
]
output_names = [
    engine.get_tensor_name(index)
    for index in range(engine.num_io_tensors)
    if engine.get_tensor_mode(engine.get_tensor_name(index)) == trt.TensorIOMode.OUTPUT
]
if len(input_names) != 1 or len(output_names) != 1:
    raise RuntimeError("预期一个输入和一个输出 tensor")
input_name = input_names[0]
output_name = output_names[0]
torch_dtypes = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16}
input_dtype = torch_dtypes[engine.get_tensor_dtype(input_name)]
output_dtype = torch_dtypes[engine.get_tensor_dtype(output_name)]
if any(dimension < 0 for dimension in engine.get_tensor_shape(input_name)):
    if not context.set_input_shape(input_name, tuple(input_tensor.shape)):
        raise RuntimeError("TensorRT 无法设置输入 shape")
trt_input = input_tensor.to(dtype=input_dtype).contiguous()
trt_output = torch.empty(tuple(context.get_tensor_shape(output_name)), dtype=output_dtype, device="cuda")
stream = torch.cuda.Stream()
if not context.set_tensor_address(input_name, int(trt_input.data_ptr())):
    raise RuntimeError("TensorRT 输入地址绑定失败")
if not context.set_tensor_address(output_name, int(trt_output.data_ptr())):
    raise RuntimeError("TensorRT 输出地址绑定失败")

for _ in range(20):
    if not context.execute_async_v3(stream.cuda_stream):
        raise RuntimeError("TensorRT warmup 失败")
stream.synchronize()

layer_profiler = LayerProfiler()
context.profiler = layer_profiler
context.enqueue_emits_profile = False
profiled_runs = []
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=2, warmup=2, active=16),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(
        str(profile_dir), worker_name="TensorRT_Engine_Bench"
    ),
) as profiler:
    for iteration in range(20):
        layer_profiler.layer_times = {}
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("第 {} 次 TensorRT profile inference 失败".format(iteration + 1))
        stream.synchronize()
        context.report_to_profiler()
        if not layer_profiler.layer_times:
            raise RuntimeError("TensorRT profiler 没有返回逐层耗时")
        profiled_runs.append(dict(layer_profiler.layer_times))
        profiler.step()

layer_samples = {}
for layer_times in profiled_runs:
    for layer_name, milliseconds in layer_times.items():
        layer_samples.setdefault(layer_name, []).append(milliseconds)
layer_summary = []
writer = SummaryWriter(str(layer_log_dir))
for layer_name, samples in layer_samples.items():
    mean_ms = float(np.mean(samples))
    median_ms = float(np.median(samples))
    layer_summary.append({"name": layer_name, "mean_ms": mean_ms, "median_ms": median_ms})
    for iteration, milliseconds in enumerate(samples):
        writer.add_scalar("LayerTime_ms/" + layer_name, milliseconds, iteration)
    writer.add_scalar("MedianLayerTime_ms/" + layer_name, median_ms, 0)
layer_summary.sort(key=lambda item: item["median_ms"], reverse=True)
summary_table = ["| Rank | TensorRT layer | Median time (ms) |", "| ---: | --- | ---: |"]
for rank, item in enumerate(layer_summary, start=1):
    summary_table.append("| {} | {} | {:.6f} |".format(rank, item["name"].replace("|", "\\|"), item["median_ms"]))
writer.add_text("LayerSummary/SortedByMedianTime", "\n".join(summary_table), 0)
writer.flush()
writer.close()

output_path = profile_dir / "layer_times.json"
output_path.write_text(json.dumps(layer_summary, indent=2) + "\n")
print("TensorRT layer profile（按中位耗时降序，单位 ms）：")
for rank, item in enumerate(layer_summary, start=1):
    print("{:3d}. {:<72s} median={:.6f} mean={:.6f}".format(
        rank, item["name"], item["median_ms"], item["mean_ms"]
    ))
print("Layer JSON：{}".format(output_path))
print("TensorBoard：tensorboard --logdir={}".format(profile_dir))
