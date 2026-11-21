"""Compare eager TensorRT submission with CUDA Graph replay on dog.jpg."""

import json
import shutil
from pathlib import Path
from time import perf_counter

import numpy as np
import tensorrt as trt
import torch
from PIL import Image
from torchvision.models import MobileNet_V3_Small_Weights

project_dir = Path(__file__).resolve().parent
model_dir = project_dir.parent.parent
source_engine = model_dir / "04_model_optimize" / "baseline" / "engine" / "mobilenet_v3_small_fp16.engine"
engine_path = project_dir / "engine" / source_engine.name
image_path = model_dir / "04_model_optimize" / "baseline" / "dog.jpg"
metrics_dir = project_dir / "metrics"
metrics_dir.mkdir(parents=True, exist_ok=True)
output_path = metrics_dir / "cudagraph_latency.json"
if not engine_path.is_file():
    shutil.copy2(source_engine, engine_path)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA Graph benchmark 需要 CUDA GPU")
weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
input_tensor = weights.transforms()(Image.open(image_path).convert("RGB")).unsqueeze(0).cuda()

runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
if engine is None:
    raise RuntimeError("TensorRT 无法加载 engine：{}".format(engine_path))
context = engine.create_execution_context()
if context is None:
    raise RuntimeError("TensorRT 无法创建 execution context")

input_names = []
output_names = []
for index in range(engine.num_io_tensors):
    name = engine.get_tensor_name(index)
    if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
        input_names.append(name)
    else:
        output_names.append(name)
if len(input_names) != 1 or len(output_names) != 1:
    raise RuntimeError("预期 engine 恰有一个输入和一个输出 tensor")
input_name = input_names[0]
output_name = output_names[0]
input_shape = tuple(engine.get_tensor_shape(input_name))
if any(dimension < 0 for dimension in input_shape):
    if not context.set_input_shape(input_name, tuple(input_tensor.shape)):
        raise RuntimeError("TensorRT 无法设置输入 shape")
elif input_shape != tuple(input_tensor.shape):
    raise RuntimeError("engine 输入 {} 与图片输入 {} 不一致".format(input_shape, tuple(input_tensor.shape)))

torch_dtypes = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16}
input_dtype = torch_dtypes[engine.get_tensor_dtype(input_name)]
output_dtype = torch_dtypes[engine.get_tensor_dtype(output_name)]
trt_input = input_tensor.to(dtype=input_dtype).contiguous()
trt_output = torch.empty(tuple(context.get_tensor_shape(output_name)), dtype=output_dtype, device="cuda")
stream = torch.cuda.Stream()
stream.wait_stream(torch.cuda.current_stream())
if not context.set_tensor_address(input_name, trt_input.data_ptr()):
    raise RuntimeError("TensorRT 绑定输入 tensor 失败")
if not context.set_tensor_address(output_name, trt_output.data_ptr()):
    raise RuntimeError("TensorRT 绑定输出 tensor 失败")

for _ in range(50):
    if not context.execute_async_v3(stream.cuda_stream):
        raise RuntimeError("TensorRT warmup 执行失败")
torch.cuda.synchronize()

graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph, stream=stream):
    if not context.execute_async_v3(stream.cuda_stream):
        raise RuntimeError("CUDA Graph capture 时 TensorRT 执行失败")
torch.cuda.synchronize()

rounds = []
for repeat in range(3):
    for mode in ["eager", "cuda_graph"]:
        for _ in range(20):
            if mode == "eager":
                if not context.execute_async_v3(stream.cuda_stream):
                    raise RuntimeError("TensorRT eager warmup 失败")
            else:
                with torch.cuda.stream(stream):
                    graph.replay()
        stream.synchronize()

        samples_ms = []
        for _ in range(200):
            started_at = perf_counter()
            if mode == "eager":
                if not context.execute_async_v3(stream.cuda_stream):
                    raise RuntimeError("TensorRT eager inference 失败")
            else:
                with torch.cuda.stream(stream):
                    graph.replay()
            stream.synchronize()
            samples_ms.append((perf_counter() - started_at) * 1000.0)

        result = {
            "repeat": repeat + 1,
            "mode": mode,
            "mean_ms": float(np.mean(samples_ms)),
            "median_ms": float(np.median(samples_ms)),
            "p90_ms": float(np.percentile(samples_ms, 90)),
            "p95_ms": float(np.percentile(samples_ms, 95)),
            "p99_ms": float(np.percentile(samples_ms, 99)),
            "cv_percent": float(np.std(samples_ms) / np.mean(samples_ms) * 100.0),
        }
        rounds.append(result)
        print(
            "run={} mode={} median={:.4f} ms p95={:.4f} ms cv={:.1f}%".format(
                result["repeat"], mode, result["median_ms"], result["p95_ms"], result["cv_percent"]
            ),
            flush=True,
        )

if not context.execute_async_v3(stream.cuda_stream):
    raise RuntimeError("Eager 输出校验执行失败")
stream.synchronize()
eager_output = trt_output.float().cpu().numpy().copy()
with torch.cuda.stream(stream):
    graph.replay()
stream.synchronize()
graph_output = trt_output.float().cpu().numpy().copy()
summary = {
    "model": "MobileNetV3-Small TensorRT FP16",
    "input": str(image_path),
    "input_shape": list(input_tensor.shape),
    "batch_size": 1,
    "engine_execution_only": True,
    "warmup_per_round": 20,
    "iterations_per_round": 200,
    "rounds": rounds,
    "top1_agreement": int(eager_output[0].argmax()) == int(graph_output[0].argmax()),
    "max_abs_output_difference": float(np.abs(eager_output - graph_output).max()),
}
output_path.write_text(json.dumps(summary, indent=2) + "\n")
print("Top-1 一致：{}，最大输出绝对误差：{:.8g}".format(
    summary["top1_agreement"], summary["max_abs_output_difference"]
))
print("结果：{}".format(output_path))
