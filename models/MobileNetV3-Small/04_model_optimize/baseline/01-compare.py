# Compare PyTorch and TensorRT profiles using the same preprocessed dog.jpg input.
# Run from this directory:
#   sudo -E python 01-compare.py
#   tensorboard --logdir=profiling
import torch
import torch.profiler
import torchvision
import tensorrt as trt
from tqdm import tqdm
from PIL import Image
from torchvision.models import MobileNet_V3_Small_Weights

PROFILE_LOG_DIR = "profiling"

# load model
weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
MODEL = torchvision.models.mobilenet_v3_small(weights=None)
MODEL.load_state_dict(torch.load(
    "weights/mobilenet_v3_small-047dcff4.pth",
    map_location="cpu",
    weights_only=True,
))
input_tensor = weights.transforms()(Image.open("dog.jpg").convert("RGB")).unsqueeze(0).cuda().half()
print(f"The input shape is: {input_tensor.shape}")
SAMPLES = [input_tensor for _ in range(1024)]

# Model has to be in eval mode, and deploy to cuda.
MODEL.eval()
MODEL.cuda().half()

# benchmark with pytorch
with torch.inference_mode():
    for sample in tqdm(SAMPLES, desc='Torch Executing'):
        MODEL(sample)
        torch.cuda.synchronize()

# load the existing TensorRT engine
ENGINE_PATH = "engine/mobilenet_v3_small_fp16.engine"
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
if -1 in tuple(engine.get_tensor_shape(input_name)):
    if not context.set_input_shape(input_name, tuple(input_tensor.shape)):
        raise RuntimeError("TensorRT 无法设置输入 shape")

torch_dtypes = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
}
input_dtype = torch_dtypes[engine.get_tensor_dtype(input_name)]
output_dtype = torch_dtypes[engine.get_tensor_dtype(output_name)]
d_input = torch.empty(tuple(context.get_tensor_shape(input_name)), dtype=input_dtype, device="cuda")
d_output = torch.empty(tuple(context.get_tensor_shape(output_name)), dtype=output_dtype, device="cuda")
if not context.set_tensor_address(input_name, d_input.data_ptr()):
    raise RuntimeError("TensorRT 绑定输入 tensor 失败")
if not context.set_tensor_address(output_name, d_output.data_ptr()):
    raise RuntimeError("TensorRT 绑定输出 tensor 失败")
stream = torch.cuda.Stream()
d_input.copy_(input_tensor.to(dtype=input_dtype))
torch.cuda.synchronize()

def trt_forward():
    if not context.execute_async_v3(stream.cuda_stream):
        raise RuntimeError("TensorRT inference 执行失败")
    stream.synchronize()

for _ in tqdm(SAMPLES, desc='TRT Executing'):
    trt_forward()

# profile the TensorRT engine
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(
        wait=2,
        warmup=1,
        active=7),
    # on_trace_ready=trace_handler
    on_trace_ready=torch.profiler.tensorboard_trace_handler(
        PROFILE_LOG_DIR, worker_name='TensorRT_Engine_Compare'
    )
    # used when outputting for tensorboard
) as p:
    for iter in range(10):
        trt_forward()
        # send a signal to the profiler that the next iteration has started
        p.step()

# profile the PyTorch model
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(
        wait=2,
        warmup=1,
        active=7),
    # on_trace_ready=trace_handler
    on_trace_ready=torch.profiler.tensorboard_trace_handler(
        PROFILE_LOG_DIR, worker_name='PyTorch_Model_Compare'
    )
    # used when outputting for tensorboard
) as p:
    with torch.inference_mode():
        for iter in range(10):
            MODEL(input_tensor)
            torch.cuda.synchronize()
            # send a signal to the profiler that the next iteration has started
            p.step()
