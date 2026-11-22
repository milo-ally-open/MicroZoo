import json
from collections import defaultdict
from pathlib import Path

import cv2
import tensorrt as trt
import torch
import torch.profiler
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

project_dir = Path(__file__).resolve().parent
model_dir = project_dir.parent
checkpoint_path = model_dir / "01_model_train" / "checkpoints" / "yolo26n.pt"
image_dir = model_dir / "01_model_train" / "datasets" / "holdout" / "images" / "val"
engine_path = model_dir / "02_model_convert" / "engine" / "yolo26n_pruned_int8.engine"
profile_dir = project_dir / "profiling"
profile_dir.mkdir(parents=True, exist_ok=True)
image_paths = sorted(image_dir.glob("*.jpg"))

if not torch.cuda.is_available():
    raise RuntimeError("当前 PyTorch 无法使用 CUDA，不能执行 GPU profiling")
if not checkpoint_path.is_file() or not engine_path.is_file() or not image_paths:
    raise FileNotFoundError("请先完成 v2 权重、留出图片和 TensorRT INT8 engine 准备")

image = cv2.imread(str(image_paths[0]), cv2.IMREAD_COLOR)
if image is None:
    raise RuntimeError("无法读取 profiling 图片：{}".format(image_paths[0]))
image = LetterBox(new_shape=(640, 640), auto=False, scaleup=True)(image=image)
input_array = image[:, :, ::-1].transpose(2, 0, 1)[None].copy()
input_tensor = torch.from_numpy(input_array).cuda().float().div_(255.0)
model = YOLO(str(checkpoint_path), task="detect").model.cuda().eval()

runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
if engine is None:
    raise RuntimeError("TensorRT 无法加载 INT8 engine：{}".format(engine_path))
context = engine.create_execution_context()
if context is None:
    raise RuntimeError("TensorRT 无法创建 execution context")
input_names = [engine.get_tensor_name(index) for index in range(engine.num_io_tensors)
               if engine.get_tensor_mode(engine.get_tensor_name(index)) == trt.TensorIOMode.INPUT]
output_names = [engine.get_tensor_name(index) for index in range(engine.num_io_tensors)
                if engine.get_tensor_mode(engine.get_tensor_name(index)) == trt.TensorIOMode.OUTPUT]
if len(input_names) != 1 or len(output_names) != 1:
    raise RuntimeError("预期 engine 恰有一个输入和一个输出")
input_name, output_name = input_names[0], output_names[0]
if any(dimension < 0 for dimension in engine.get_tensor_shape(input_name)):
    if not context.set_input_shape(input_name, tuple(input_tensor.shape)):
        raise RuntimeError("TensorRT 无法设置输入 shape")
torch_types = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16}
if engine.get_tensor_dtype(input_name) not in torch_types or engine.get_tensor_dtype(output_name) not in torch_types:
    raise RuntimeError("profiling 需要 FP32/FP16 engine I/O，实际为 {} -> {}".format(
        engine.get_tensor_dtype(input_name), engine.get_tensor_dtype(output_name)
    ))
trt_input = input_tensor.to(dtype=torch_types[engine.get_tensor_dtype(input_name)]).contiguous()
trt_output = torch.empty(
    tuple(context.get_tensor_shape(output_name)),
    dtype=torch_types[engine.get_tensor_dtype(output_name)],
    device="cuda",
)
if not context.set_tensor_address(input_name, int(trt_input.data_ptr())):
    raise RuntimeError("TensorRT 输入 tensor 绑定失败")
if not context.set_tensor_address(output_name, int(trt_output.data_ptr())):
    raise RuntimeError("TensorRT 输出 tensor 绑定失败")
stream = torch.cuda.Stream()

class LayerProfiler(trt.IProfiler):
    def __init__(self):
        super().__init__()
        self.samples = defaultdict(list)

    def report_layer_time(self, layer_name, milliseconds):
        self.samples[layer_name].append(float(milliseconds))

profiler = LayerProfiler()
context.profiler = profiler
context.enqueue_emits_profile = False
torch.cuda.synchronize()
for _ in range(10):
    if not context.execute_async_v3(stream.cuda_stream):
        raise RuntimeError("TensorRT INT8 warmup inference 失败")
    stream.synchronize()
    context.report_to_profiler()

profiler.samples.clear()
for _ in range(10):
    if not context.execute_async_v3(stream.cuda_stream):
        raise RuntimeError("TensorRT INT8 layer profiling inference 失败")
    stream.synchronize()
    context.report_to_profiler()
layer_rows = [
    {"layer": name, "mean_ms": sum(times) / len(times), "samples": len(times)}
    for name, times in profiler.samples.items()
]
layer_rows.sort(key=lambda row: row["mean_ms"], reverse=True)
layer_profile_path = profile_dir / "tensorrt_int8_layers.json"
layer_profile_path.write_text(json.dumps({
    "engine": str(engine_path.relative_to(model_dir)),
    "tensorrt_version": trt.__version__,
    "warmup_iterations": 10,
    "profile_iterations": 10,
    "layers": layer_rows,
}, indent=2) + "\n")

# TensorBoard trace 中分别用固定的 worker 名标出原始 PyTorch 与量化 TensorRT。
context.profiler = None
context.enqueue_emits_profile = True
supported_activities = torch.profiler.supported_activities()
activities = [torch.profiler.ProfilerActivity.CPU]
if torch.profiler.ProfilerActivity.CUDA in supported_activities:
    activities.append(torch.profiler.ProfilerActivity.CUDA)
else:
    print("警告：此 PyTorch profiler 没有 CUDA activity 支持。", flush=True)

with torch.profiler.profile(
    activities=activities,
    schedule=torch.profiler.schedule(wait=2, warmup=1, active=7),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(str(profile_dir), worker_name="TensorRT_INT8"),
) as trace:
    for _ in range(10):
        with torch.profiler.record_function("TensorRT INT8 inference"):
            if not context.execute_async_v3(stream.cuda_stream):
                raise RuntimeError("TensorRT INT8 inference 失败")
            stream.synchronize()
        trace.step()
print(trace.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))

with torch.profiler.profile(
    activities=activities,
    schedule=torch.profiler.schedule(wait=2, warmup=1, active=7),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(str(profile_dir), worker_name="PyTorch_FP32"),
) as trace:
    with torch.inference_mode():
        for _ in range(10):
            with torch.profiler.record_function("PyTorch FP32 inference"):
                model(input_tensor)
                torch.cuda.synchronize()
            trace.step()
print(trace.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))

print("输入图片：{}；shape：{}".format(image_paths[0], tuple(input_tensor.shape)))
print("TensorRT INT8 layer profile：{}".format(layer_profile_path.resolve()))
print("TensorBoard worker：TensorRT_INT8、PyTorch_FP32")
print("查看命令：tensorboard --logdir={}".format(profile_dir.resolve()))
