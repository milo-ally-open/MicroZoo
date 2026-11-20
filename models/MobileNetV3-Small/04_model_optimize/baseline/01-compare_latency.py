"""在同一设备、同一预处理输入上比较 PyTorch 与 TensorRT 单样本延迟。"""

from pathlib import Path
from time import perf_counter

import numpy as np
import tensorrt as trt
import torch
from PIL import Image
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

project_dir = Path(__file__).resolve().parent
checkpoint_path = project_dir / "weights" / "mobilenet_v3_small-047dcff4.pth"
engine_path = project_dir / "engine" / "mobilenet_v3_small_fp16.engine"
image_path = project_dir / "dog.jpg"
device = torch.device("cuda")
weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
warmup_iterations = 20
measurement_iterations = 100

model = mobilenet_v3_small(weights=None)
model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
model = model.to(device).eval()

input_cpu = weights.transforms()(Image.open(image_path).convert("RGB")).unsqueeze(0)
input_tensor = input_cpu.to(device)

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
    context.set_input_shape(input_name, (1, 3, 224, 224))
output_shape = tuple(context.get_tensor_shape(output_name))
trt_logits = torch.empty(output_shape, dtype=output_dtype, device=device)
trt_input = input_tensor.to(dtype=input_dtype).contiguous()
stream = torch.cuda.Stream(device=device)
context.set_tensor_address(input_name, int(trt_input.data_ptr()))
context.set_tensor_address(output_name, int(trt_logits.data_ptr()))

with torch.inference_mode():
    for _ in range(warmup_iterations):
        model(input_tensor)
torch.cuda.synchronize(device)

torch.cuda.current_stream(device).synchronize()
for _ in range(warmup_iterations):
    if not context.execute_async_v3(stream.cuda_stream):
        raise RuntimeError("TensorRT warmup inference 执行失败")
stream.synchronize()

pytorch_times = []
with torch.inference_mode():
    for _ in range(measurement_iterations):
        torch.cuda.synchronize(device)
        started_at = perf_counter()
        model(input_tensor)
        torch.cuda.synchronize(device)
        pytorch_times.append((perf_counter() - started_at) * 1000)

tensorrt_times = []
for _ in range(measurement_iterations):
    torch.cuda.current_stream(device).synchronize()
    started_at = perf_counter()
    if not context.execute_async_v3(stream.cuda_stream):
        raise RuntimeError("TensorRT inference 执行失败")
    stream.synchronize()
    tensorrt_times.append((perf_counter() - started_at) * 1000)

print("输入：{}，shape={}，batch=1".format(image_path, tuple(input_tensor.shape)))
print("预热次数：{}；测量次数：{}".format(warmup_iterations, measurement_iterations))
print("单样本推理延迟（含 CUDA 同步，不含图像预处理和输入传输）：")
for runtime_name, times in [("PyTorch", pytorch_times), ("TensorRT", tensorrt_times)]:
    print("{}：mean {:.3f} ms，P50 {:.3f} ms，P95 {:.3f} ms".format(
        runtime_name,
        float(np.mean(times)),
        float(np.percentile(times, 50)),
        float(np.percentile(times, 95)),
    ))
