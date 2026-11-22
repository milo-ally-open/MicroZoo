# Compare PyTorch and TensorRT profiler traces with the same preprocessed dog.jpg input.
# Run from this directory:
#   sudo -E python 01-profile.py
#   tensorboard --logdir=profiling
# Dependencies: torch (includes torch.profiler), torch-tb-profiler, tensorboard,
#               torchvision, Pillow, TensorRT.
from pathlib import Path

import tensorrt as trt
import torch
import torch.profiler
import torchvision
from PIL import Image
from torchvision.models import MobileNet_V3_Small_Weights

project_dir = Path(__file__).resolve().parent
model_dir = project_dir.parent
checkpoint_path = model_dir / "01_model_train" / "checkpoints" / "mobilenet_v3_small-047dcff4.pth"
image_path = model_dir / "03_model_validate" / "dog.jpg"
engine_path = model_dir / "02_model_convert" / "engine" / "mobilenet_v3_small_fp16.engine"
profile_dir = project_dir / "profiling"
profile_dir.mkdir(parents=True, exist_ok=True)

weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
input_tensor = weights.transforms()(Image.open(image_path).convert("RGB")).unsqueeze(0).cuda().half()
model = torchvision.models.mobilenet_v3_small(weights=None)
model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
model = model.cuda().half().eval()

runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
if engine is None:
    raise RuntimeError("TensorRT 无法加载 engine：{}".format(engine_path))
context = engine.create_execution_context()
if context is None:
    raise RuntimeError("TensorRT 无法创建 execution context")

input_names = [engine.get_tensor_name(index) for index in range(engine.num_io_tensors)
               if engine.get_tensor_mode(engine.get_tensor_name(index)) == trt.TensorIOMode.INPUT]
output_names = [engine.get_tensor_name(index) for index in range(engine.num_io_tensors)
                if engine.get_tensor_mode(engine.get_tensor_name(index)) == trt.TensorIOMode.OUTPUT]
if len(input_names) != 1 or len(output_names) != 1:
    raise RuntimeError("预期 engine 恰有一个输入和一个输出 tensor")

input_name = input_names[0]
output_name = output_names[0]
if any(dimension < 0 for dimension in engine.get_tensor_shape(input_name)):
    if not context.set_input_shape(input_name, tuple(input_tensor.shape)):
        raise RuntimeError("TensorRT 无法设置输入 shape")
torch_dtypes = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16}
input_dtype = torch_dtypes[engine.get_tensor_dtype(input_name)]
output_dtype = torch_dtypes[engine.get_tensor_dtype(output_name)]
trt_input = torch.empty(tuple(context.get_tensor_shape(input_name)), dtype=input_dtype, device="cuda")
trt_output = torch.empty(tuple(context.get_tensor_shape(output_name)), dtype=output_dtype, device="cuda")
trt_input.copy_(input_tensor.to(dtype=input_dtype))
if not context.set_tensor_address(input_name, int(trt_input.data_ptr())):
    raise RuntimeError("TensorRT 输入 tensor 绑定失败")
if not context.set_tensor_address(output_name, int(trt_output.data_ptr())):
    raise RuntimeError("TensorRT 输出 tensor 绑定失败")
stream = torch.cuda.Stream()
torch.cuda.synchronize()

for _ in range(10):
    if not context.execute_async_v3(stream.cuda_stream):
        raise RuntimeError("TensorRT inference 执行失败")
    stream.synchronize()
with torch.inference_mode():
    for _ in range(10):
        model(input_tensor)
        torch.cuda.synchronize()

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=2, warmup=1, active=7),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(str(profile_dir), worker_name="TensorRT_Engine"),
) as profiler:
    for _ in range(10):
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT inference 执行失败")
        stream.synchronize()
        profiler.step()

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=2, warmup=1, active=7),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(str(profile_dir), worker_name="PyTorch_Model"),
) as profiler:
    with torch.inference_mode():
        for _ in range(10):
            model(input_tensor)
            torch.cuda.synchronize()
            profiler.step()

print("模型：MobileNetV3-Small FP16 PyTorch 与 TensorRT")
print("输入图像：{}；输入 shape：{}".format(image_path, tuple(input_tensor.shape)))
print("Profiler trace：{}".format(profile_dir.resolve()))
print("TensorBoard worker：TensorRT_Engine、PyTorch_Model")
print("查看命令：tensorboard --logdir=profiling")
