"""在同一张 COCO val 图片上记录 PyTorch CUDA 与 TensorRT 的 TensorBoard trace。"""

import shutil
from pathlib import Path

import cv2
import tensorrt as trt
import torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

project_dir = Path(__file__).resolve().parent
model_dir = project_dir.parent.parent
source_engine = model_dir / "02_model_convert" / "engine" / "yolo26n_fp16.engine"
engine_path = project_dir / "engine" / source_engine.name
checkpoint_path = project_dir / "weights" / "yolo26n.pt"
image_path = project_dir / "sample.jpg"
profile_dir = project_dir / "profiling"
profile_dir.mkdir(parents=True, exist_ok=True)
engine_path.parent.mkdir(parents=True, exist_ok=True)
checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
if not engine_path.is_file():
    shutil.copy2(source_engine, engine_path)
if not checkpoint_path.is_file():
    shutil.copy2(model_dir / "01_model_train" / "checkpoints" / "yolo26n.pt", checkpoint_path)
if not torch.cuda.is_available():
    raise RuntimeError("此 profile 要求 PyTorch CUDA 可用")

image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
if image is None:
    raise FileNotFoundError("无法读取 profiler 图片：{}".format(image_path))
image = LetterBox(new_shape=(640, 640), auto=False, scaleup=True)(image=image)
input_array = image[:, :, ::-1].transpose(2, 0, 1)[None].copy()
input_tensor = torch.from_numpy(input_array).cuda().float().div_(255.0)

yolo_model = YOLO(str(checkpoint_path))
pytorch_model = yolo_model.model.cuda().half().eval()
pytorch_input = input_tensor.half()

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
    raise RuntimeError("预期一个输入和一个输出 tensor")
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
context.set_tensor_address(input_name, int(trt_input.data_ptr()))
context.set_tensor_address(output_name, int(trt_output.data_ptr()))

with torch.inference_mode():
    for _ in range(20):
        pytorch_model(pytorch_input)
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT warmup 失败")
    torch.cuda.synchronize()

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=2, warmup=1, active=7),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(str(profile_dir), worker_name="TensorRT_Engine"),
) as profiler:
    for _ in range(10):
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT profile inference 失败")
        stream.synchronize()
        profiler.step()

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=2, warmup=1, active=7),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(str(profile_dir), worker_name="PyTorch_Model"),
) as profiler:
    with torch.inference_mode():
        for _ in range(10):
            pytorch_model(pytorch_input)
            torch.cuda.synchronize()
            profiler.step()

print("PyTorch CUDA 和 TensorRT Engine trace 已写入：{}".format(profile_dir))
print("TensorBoard：tensorboard --logdir={}".format(profile_dir))
