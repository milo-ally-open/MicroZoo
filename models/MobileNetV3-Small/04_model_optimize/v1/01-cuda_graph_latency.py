"""比较普通 TensorRT enqueue 与 CUDA Graph 重放的单样本延迟。"""

from pathlib import Path
from time import perf_counter

import numpy as np
import tensorrt as trt
import torch
from PIL import Image
from torchvision.models import MobileNet_V3_Small_Weights

v1_dir = Path(__file__).resolve().parent
baseline_dir = v1_dir.parent / "baseline"
engine_path = baseline_dir / "engine" / "mobilenet_v3_small_fp16.engine"
image_path = baseline_dir / "dog.jpg"
csv_path = v1_dir / "01-cuda-graph-latency.csv"
device = torch.device("cuda")
warmup_iterations = 50
rounds = 5
iterations_per_round = 100
weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1

input_tensor = weights.transforms()(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
if engine is None:
    raise RuntimeError("TensorRT 无法加载 engine：{}".format(engine_path))

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
    raise RuntimeError("预期 TensorRT engine 有一个输入和一个输出")

input_name = input_names[0]
output_name = output_names[0]
torch_dtypes = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
}
input_dtype = torch_dtypes[engine.get_tensor_dtype(input_name)]
output_dtype = torch_dtypes[engine.get_tensor_dtype(output_name)]
if -1 in tuple(engine.get_tensor_shape(input_name)):
    context.set_input_shape(input_name, tuple(input_tensor.shape))
output_shape = tuple(context.get_tensor_shape(output_name))
trt_input = input_tensor.to(dtype=input_dtype).contiguous()
trt_output = torch.empty(output_shape, dtype=output_dtype, device=device)
stream = torch.cuda.Stream(device=device)
context.set_tensor_address(input_name, int(trt_input.data_ptr()))
context.set_tensor_address(output_name, int(trt_output.data_ptr()))

torch.cuda.current_stream(device).synchronize()
for _ in range(warmup_iterations):
    if not context.execute_async_v3(stream.cuda_stream):
        raise RuntimeError("TensorRT warmup inference 执行失败")
stream.synchronize()

cuda_graph = torch.cuda.CUDAGraph()
stream.wait_stream(torch.cuda.current_stream(device))
with torch.cuda.graph(cuda_graph, stream=stream):
    if not context.execute_async_v3(stream.cuda_stream):
        raise RuntimeError("CUDA Graph capture 中的 TensorRT inference 执行失败")
stream.synchronize()

for _ in range(warmup_iterations):
    cuda_graph.replay()
torch.cuda.synchronize(device)

original_input = trt_input.clone()
trt_input.zero_()
torch.cuda.synchronize(device)
cuda_graph.replay()
torch.cuda.synchronize(device)
graph_zero_output = trt_output.clone()

trt_input.copy_(original_input)
torch.cuda.synchronize(device)
if not context.execute_async_v3(stream.cuda_stream):
    raise RuntimeError("TensorRT reference inference 执行失败")
stream.synchronize()
eager_image_output = trt_output.clone()

trt_input.zero_()
torch.cuda.synchronize(device)
if not context.execute_async_v3(stream.cuda_stream):
    raise RuntimeError("TensorRT zero-input inference 执行失败")
stream.synchronize()
eager_zero_output = trt_output.clone()
if not torch.allclose(graph_zero_output, eager_zero_output, rtol=1e-4, atol=1e-5):
    raise RuntimeError("CUDA Graph 输出与普通 TensorRT 输出不一致")

trt_input.copy_(original_input)
torch.cuda.synchronize(device)
cuda_graph.replay()
torch.cuda.synchronize(device)
graph_image_output = trt_output.clone()
if not torch.allclose(graph_image_output, eager_image_output, rtol=1e-4, atol=1e-5):
    raise RuntimeError("CUDA Graph 与普通 TensorRT 对狗图输入的输出不一致")

eager_times = []
graph_times = []
with csv_path.open("w", encoding="utf-8") as csv_file:
    csv_file.write("mode,round,iteration,latency_ms\n")
    for round_index in range(rounds):
        for iteration_index in range(iterations_per_round):
            torch.cuda.synchronize(device)
            started_at = perf_counter()
            if not context.execute_async_v3(stream.cuda_stream):
                raise RuntimeError("TensorRT inference 执行失败")
            torch.cuda.synchronize(device)
            latency_ms = (perf_counter() - started_at) * 1000
            eager_times.append(latency_ms)
            csv_file.write("eager,{},{},{:.6f}\n".format(round_index + 1, iteration_index + 1, latency_ms))

        for iteration_index in range(iterations_per_round):
            torch.cuda.synchronize(device)
            started_at = perf_counter()
            cuda_graph.replay()
            torch.cuda.synchronize(device)
            latency_ms = (perf_counter() - started_at) * 1000
            graph_times.append(latency_ms)
            csv_file.write("cuda_graph,{},{},{:.6f}\n".format(round_index + 1, iteration_index + 1, latency_ms))

print("Engine：{}".format(engine_path))
print("输入：{}，shape={}，batch=1".format(image_path, tuple(input_tensor.shape)))
print("预热：{} 次；{} 轮 × 每轮 {} 次；不含图像预处理和输入传输。".format(
    warmup_iterations, rounds, iterations_per_round,
))
print("计时方式：每次推理前后同步 CUDA；每轮先测 eager，再测 CUDA Graph。")
print("正确性检查：CUDA Graph 与普通 TensorRT 对零输入和狗图输入的输出一致。")
print("Latency：mean / P50 / P95（ms）")
for mode, times in [("TensorRT eager", eager_times), ("CUDA Graph", graph_times)]:
    print("{}：{:.3f} / {:.3f} / {:.3f}".format(
        mode,
        float(np.mean(times)),
        float(np.percentile(times, 50)),
        float(np.percentile(times, 95)),
    ))
print("CUDA Graph 对比 eager 的 P50 变化：{:+.1f}%".format(
    (np.percentile(graph_times, 50) / np.percentile(eager_times, 50) - 1) * 100,
))
print("逐次延迟数据：{}".format(csv_path))
