"""Compare TensorRT eager enqueue and CUDA Graph replay on the fixed COCO sample."""

import json
import shutil
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import tensorrt as trt
import torch
from ultralytics.data.augment import LetterBox

project_dir = Path(__file__).resolve().parent
model_dir = project_dir.parent.parent
source_engine = model_dir / "02_model_convert" / "engine" / "yolo26n_fp16.engine"
engine_path = project_dir / "engine" / source_engine.name
image_path = model_dir / "04_model_optimize" / "baseline" / "sample.jpg"
metrics_dir = project_dir / "metrics"
metrics_dir.mkdir(parents=True, exist_ok=True)
engine_path.parent.mkdir(parents=True, exist_ok=True)
output_path = metrics_dir / "cudagraph_latency.json"
if not engine_path.is_file():
    shutil.copy2(source_engine, engine_path)

image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
if image is None:
    raise FileNotFoundError("无法读取固定输入图片：{}".format(image_path))
image = LetterBox(new_shape=(640, 640), auto=False, scaleup=True)(image=image)
input_array = image[:, :, ::-1].transpose(2, 0, 1)[None].copy()
input_tensor = torch.from_numpy(input_array).cuda().float().div_(255.0)

runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
if engine is None:
    raise RuntimeError("TensorRT 无法加载 engine")
context = engine.create_execution_context()
input_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)
               if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT]
output_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)
                if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT]
if len(input_names) != 1 or len(output_names) != 1:
    raise RuntimeError("预期 engine 有一个输入和一个输出 tensor")
input_name, output_name = input_names[0], output_names[0]
trt_dtypes = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16}
input_dtype = trt_dtypes[engine.get_tensor_dtype(input_name)]
output_dtype = trt_dtypes[engine.get_tensor_dtype(output_name)]
if any(dimension < 0 for dimension in engine.get_tensor_shape(input_name)):
    if not context.set_input_shape(input_name, tuple(input_tensor.shape)):
        raise RuntimeError("TensorRT 无法设置输入 shape")
trt_input = input_tensor.to(dtype=input_dtype).contiguous()
trt_output = torch.empty(tuple(context.get_tensor_shape(output_name)), dtype=output_dtype, device="cuda")
stream = torch.cuda.Stream()
stream.wait_stream(torch.cuda.current_stream())
if not context.set_tensor_address(input_name, int(trt_input.data_ptr())):
    raise RuntimeError("TensorRT 输入地址绑定失败")
if not context.set_tensor_address(output_name, int(trt_output.data_ptr())):
    raise RuntimeError("TensorRT 输出地址绑定失败")

for _ in range(50):
    if not context.execute_async_v3(stream.cuda_stream):
        raise RuntimeError("TensorRT warmup 失败")
stream.synchronize()

graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph, stream=stream):
    if not context.execute_async_v3(stream.cuda_stream):
        raise RuntimeError("CUDA Graph capture 时 TensorRT 执行失败")
torch.cuda.synchronize()

# 对固定 COCO 子集逐张比较 eager 与 graph 的原始 engine 输出。
dataset_dir = model_dir / "01_model_train" / "datasets" / "coco-val-32"
manifest_path = dataset_dir / "selection.txt"
image_dir = dataset_dir / "images" / "val"
if not manifest_path.is_file():
    raise FileNotFoundError("先运行 01_model_train/00-prepare_coco_val.py")
image_names = [
    line.split()[1]
    for line in manifest_path.read_text().splitlines()
    if line.strip() and not line.startswith("#")
]
if len(image_names) != 32:
    raise RuntimeError("固定 COCO 验证子集应有 32 张图片，当前为 {} 张".format(len(image_names)))
comparison_max_abs = 0.0
comparison_nonidentical_images = 0
for image_name in image_names:
    frame = cv2.imread(str(image_dir / image_name), cv2.IMREAD_COLOR)
    if frame is None:
        raise FileNotFoundError("无法读取 COCO 验证图片：{}".format(image_dir / image_name))
    frame = LetterBox(new_shape=(640, 640), auto=False, scaleup=True)(image=frame)
    array = frame[:, :, ::-1].transpose(2, 0, 1)[None].copy().astype(np.float32) / 255.0
    validation_input = torch.from_numpy(array).to(device="cuda", dtype=input_dtype)
    torch.cuda.synchronize()
    trt_input.copy_(validation_input)
    torch.cuda.synchronize()

    if not context.execute_async_v3(stream.cuda_stream):
        raise RuntimeError("TensorRT eager 数值比较执行失败：{}".format(image_name))
    stream.synchronize()
    eager_output = trt_output.float().cpu().numpy().copy()
    with torch.cuda.stream(stream):
        graph.replay()
    stream.synchronize()
    graph_output = trt_output.float().cpu().numpy().copy()
    image_max_abs = float(np.abs(eager_output - graph_output).max())
    comparison_max_abs = max(comparison_max_abs, image_max_abs)
    if not np.array_equal(eager_output, graph_output):
        comparison_nonidentical_images += 1

# 性能计时固定使用同一张 sample.jpg，图片传输不计入 engine 延迟。
trt_input.copy_(input_tensor.to(dtype=input_dtype))
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
        rounds.append({
            "repeat": repeat + 1,
            "mode": mode,
            "mean_ms": float(np.mean(samples_ms)),
            "median_ms": float(np.median(samples_ms)),
            "p90_ms": float(np.percentile(samples_ms, 90)),
            "p95_ms": float(np.percentile(samples_ms, 95)),
            "p99_ms": float(np.percentile(samples_ms, 99)),
            "cv_percent": float(np.std(samples_ms) / np.mean(samples_ms) * 100.0),
        })
        print("run={} mode={} median={:.4f} ms p95={:.4f} ms cv={:.1f}%".format(
            repeat + 1,
            mode,
            rounds[-1]["median_ms"],
            rounds[-1]["p95_ms"],
            rounds[-1]["cv_percent"],
        ), flush=True)

if not context.execute_async_v3(stream.cuda_stream):
    raise RuntimeError("TensorRT eager output check failed")
stream.synchronize()
eager_output = trt_output.float().cpu().numpy().copy()
with torch.cuda.stream(stream):
    graph.replay()
stream.synchronize()
graph_output = trt_output.float().cpu().numpy()
summary = {
    "model": "YOLO26n TensorRT FP16",
    "input_image": str(image_path),
    "engine_execution_only": True,
    "input_shape": list(input_tensor.shape),
    "warmup_per_round": 20,
    "iterations_per_round": 200,
    "rounds": rounds,
    "output_shape": list(eager_output.shape),
    "numerical_validation": {
        "dataset": "COCO val2017 deterministic 32-image subset",
        "image_count": len(image_names),
        "nonidentical_image_count": comparison_nonidentical_images,
        "max_abs_output_difference": comparison_max_abs,
        "bitwise_equal": comparison_nonidentical_images == 0 and comparison_max_abs == 0.0,
    },
}
output_path.write_text(json.dumps(summary, indent=2) + "\n")
print("结果：{}".format(output_path))
