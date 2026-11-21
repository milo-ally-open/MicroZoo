# Use the real dog.jpg preprocessing and profile TensorRT only.
# Run from this directory:
#   sudo -E python 02-bench.py
#   tensorboard --logdir=profiling

import tensorrt as trt
import torch
import torch.profiler
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import MobileNet_V3_Small_Weights


class LayerProfiler(trt.IProfiler):
    def __init__(self):
        trt.IProfiler.__init__(self)
        self.layer_times = {}

    def report_layer_time(self, layer_name, ms):
        self.layer_times[layer_name] = float(ms)


ENGINE_PATH = "engine/mobilenet_v3_small_fp16.engine"
IMAGE_PATH = "dog.jpg"
PROFILE_LOG_DIR = "profiling"
LAYER_LOG_DIR = "profiling/TensorRT_Bench"
WARMUP_ITERATIONS = 10
PROFILE_ITERATIONS = 20

weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
input_tensor = weights.transforms()(
    Image.open(IMAGE_PATH).convert("RGB")
).unsqueeze(0).cuda().half()
print("实际输入图片：{}".format(IMAGE_PATH))
print("预处理后输入 shape：{}".format(tuple(input_tensor.shape)))

logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)
with open(ENGINE_PATH, "rb") as engine_file:
    engine = runtime.deserialize_cuda_engine(engine_file.read())
if engine is None:
    raise RuntimeError("TensorRT 无法加载 engine：{}".format(ENGINE_PATH))

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
engine_input_shape = tuple(engine.get_tensor_shape(input_name))
if any(dimension < 0 for dimension in engine_input_shape):
    if not context.set_input_shape(input_name, tuple(input_tensor.shape)):
        raise RuntimeError(
            "TensorRT 无法设置输入 shape：{}".format(tuple(input_tensor.shape))
        )
elif engine_input_shape != tuple(input_tensor.shape):
    raise RuntimeError(
        "engine 输入 shape {} 与图片预处理结果 {} 不一致".format(
            engine_input_shape, tuple(input_tensor.shape)
        )
    )

torch_dtypes = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
}
input_dtype = torch_dtypes[engine.get_tensor_dtype(input_name)]
output_dtype = torch_dtypes[engine.get_tensor_dtype(output_name)]
d_input = torch.empty(
    tuple(context.get_tensor_shape(input_name)), dtype=input_dtype, device="cuda"
)
d_output = torch.empty(
    tuple(context.get_tensor_shape(output_name)), dtype=output_dtype, device="cuda"
)
# Prepare the real preprocessed image once; the profiled loop measures the engine only.
d_input.copy_(input_tensor.to(dtype=input_dtype))
if not context.set_tensor_address(input_name, d_input.data_ptr()):
    raise RuntimeError("TensorRT 绑定输入 tensor 失败")
if not context.set_tensor_address(output_name, d_output.data_ptr()):
    raise RuntimeError("TensorRT 绑定输出 tensor 失败")

stream = torch.cuda.Stream()
torch.cuda.synchronize()
for _ in range(WARMUP_ITERATIONS):
    if not context.execute_async_v3(stream.cuda_stream):
        raise RuntimeError("TensorRT warmup 执行失败")
stream.synchronize()

layer_profiler = LayerProfiler()
context.profiler = layer_profiler
context.enqueue_emits_profile = False
profiled_runs = []
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(wait=2, warmup=2, active=16),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(
        PROFILE_LOG_DIR, worker_name="TensorRT_Engine_Bench"
    ),
) as profiler:
    for iteration in range(PROFILE_ITERATIONS):
        layer_profiler.layer_times = {}
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT 第 {} 次 profile 执行失败".format(iteration + 1))
        stream.synchronize()
        if not context.report_to_profiler():
            raise RuntimeError("TensorRT 未能生成逐层 profile")
        if not layer_profiler.layer_times:
            raise RuntimeError("TensorRT profiler 没有返回逐层耗时")
        profiled_runs.append(dict(layer_profiler.layer_times))
        profiler.step()

layer_samples = {}
for layer_times in profiled_runs:
    for layer_name, time_ms in layer_times.items():
        layer_samples.setdefault(layer_name, []).append(time_ms)

layer_means = []
writer = SummaryWriter(LAYER_LOG_DIR)
for layer_name, samples in layer_samples.items():
    mean_ms = sum(samples) / len(samples)
    layer_means.append((layer_name, mean_ms))
    for iteration, time_ms in enumerate(samples):
        writer.add_scalar("LayerTime_ms/" + layer_name, time_ms, iteration)
    writer.add_scalar("MeanLayerTime_ms/" + layer_name, mean_ms, 0)

layer_means.sort(key=lambda item: item[1], reverse=True)
table = [
    "| Rank | TensorRT layer | Mean time (ms) |",
    "| ---: | --- | ---: |",
]
for rank, (layer_name, mean_ms) in enumerate(layer_means, start=1):
    table.append("| {} | {} | {:.6f} |".format(rank, layer_name.replace("|", "\\|"), mean_ms))
writer.add_text("LayerSummary/SortedByMeanTime", "\n".join(table), 0)
writer.flush()
writer.close()

print("TensorRT layer profile（按平均耗时降序，单位 ms）：")
for rank, (layer_name, mean_ms) in enumerate(layer_means, start=1):
    print("{:3d}. {:<70s} {:.6f}".format(rank, layer_name, mean_ms))
print("\nTensorBoard 日志已写入：{} 和 {}".format(PROFILE_LOG_DIR, LAYER_LOG_DIR))
