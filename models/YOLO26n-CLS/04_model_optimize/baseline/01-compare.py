"""在同一张 sample.jpg 上生成 PyTorch 和 TensorRT 的 TensorBoard profiler trace。"""

import shutil
from pathlib import Path

import tensorrt as trt
import torch
from PIL import Image
from ultralytics import YOLO
from ultralytics.data.augment import classify_transforms

project_dir = Path(__file__).resolve().parent
model_dir = project_dir.parent.parent
source_engine = model_dir / "02_model_convert" / "engine" / "yolo26n-cls_fp16.engine"
engine_path = project_dir / "engine" / source_engine.name
checkpoint_path = project_dir / "weights" / "yolo26n-cls.pt"
image_path = project_dir / "sample.jpg"
profile_dir = project_dir / "profiling"
profile_dir.mkdir(parents=True, exist_ok=True)
if not engine_path.is_file():
    shutil.copy2(source_engine, engine_path)

if not torch.cuda.is_available():
    raise RuntimeError("此 profile 要求 PyTorch CUDA 可用")
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
transform = classify_transforms(224)
input_tensor = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).cuda()

yolo_model = YOLO(str(checkpoint_path))
pytorch_model = yolo_model.model.cuda().half().eval()
pytorch_input = input_tensor.half()

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
    if not context.set_input_shape(input_name, (1, 3, 224, 224)):
        raise RuntimeError("TensorRT 无法设置输入 shape")
trt_input = input_tensor.to(dtype=input_dtype).contiguous()
trt_output = torch.empty(tuple(context.get_tensor_shape(output_name)), dtype=output_dtype, device="cuda")
stream = torch.cuda.Stream()
if not context.set_tensor_address(input_name, int(trt_input.data_ptr())):
    raise RuntimeError("TensorRT 输入地址绑定失败")
if not context.set_tensor_address(output_name, int(trt_output.data_ptr())):
    raise RuntimeError("TensorRT 输出地址绑定失败")

with torch.inference_mode():
    for _ in range(20):
        pytorch_model(pytorch_input)
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT warmup 失败")
    torch.cuda.synchronize()

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=2, warmup=1, active=7),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(
        str(profile_dir), worker_name="TensorRT_Engine"
    ),
) as profiler:
    for _ in range(10):
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT profile inference 失败")
        stream.synchronize()
        profiler.step()

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=2, warmup=1, active=7),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(
        str(profile_dir), worker_name="PyTorch_Model"
    ),
) as profiler:
    with torch.inference_mode():
        for _ in range(10):
            pytorch_model(pytorch_input)
            torch.cuda.synchronize()
            profiler.step()

print("PyTorch 和 TensorRT 的 profiler trace 已写入：{}".format(profile_dir))
print("TensorBoard：tensorboard --logdir={}".format(profile_dir))
