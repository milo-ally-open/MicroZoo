from pathlib import Path

import numpy as np
import onnxruntime as ort
import tensorrt as trt
import torch
from PIL import Image
from torchvision.models import MobileNet_V3_Small_Weights

project_dir = Path(__file__).resolve().parent
onnx_path = project_dir / "onnx" / "mobilenet_v3_small.onnx"
engine_path = project_dir / "engine" / "mobilenet_v3_small_fp16.engine"
image_path = project_dir / "dog.jpg"
expected_label = "Samoyed"
device = torch.device("cuda")
weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
categories = weights.meta["categories"]
expected_index = categories.index(expected_label)

input_cpu = weights.transforms()(Image.open(image_path).convert("RGB")).unsqueeze(0)

onnx_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
onnx_input_name = onnx_session.get_inputs()[0].name
onnx_output_name = onnx_session.get_outputs()[0].name
onnx_logits = onnx_session.run(
    [onnx_output_name],
    {onnx_input_name: input_cpu.numpy()},
)[0]

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
trt_input = input_cpu.to(device=device, dtype=input_dtype).contiguous()
stream = torch.cuda.Stream(device=device)
context.set_tensor_address(input_name, int(trt_input.data_ptr()))
context.set_tensor_address(output_name, int(trt_logits.data_ptr()))
torch.cuda.current_stream(device).synchronize()
if not context.execute_async_v3(stream.cuda_stream):
    raise RuntimeError("TensorRT inference 执行失败")
stream.synchronize()
trt_logits = trt_logits.cpu().numpy().astype(np.float32)

print("测试图像：{}".format(image_path))
print("人工参考标签：{}".format(expected_label))
print("ONNX Runtime CPU Top-1：{}；正确={}".format(
    categories[int(np.argmax(onnx_logits[0]))],
    int(np.argmax(onnx_logits[0])) == expected_index,
))
print("ONNX Runtime CPU Top-5：{}".format([
    categories[index]
    for index in np.argsort(onnx_logits[0])[-5:][::-1].tolist()
]))
print("TensorRT FP16 Top-1：{}；正确={}".format(
    categories[int(np.argmax(trt_logits[0]))],
    int(np.argmax(trt_logits[0])) == expected_index,
))
print("TensorRT FP16 Top-5：{}".format([
    categories[index]
    for index in np.argsort(trt_logits[0])[-5:][::-1].tolist()
]))
