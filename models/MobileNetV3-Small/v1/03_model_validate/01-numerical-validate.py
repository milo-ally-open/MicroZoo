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
device = torch.device("cuda")
weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1

input_cpu = weights.transforms()(Image.open(image_path).convert("RGB")).unsqueeze(0)

onnx_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
onnx_input_name = onnx_session.get_inputs()[0].name
onnx_output_name = onnx_session.get_outputs()[0].name
onnx_logits_array = onnx_session.run(
    [onnx_output_name],
    {onnx_input_name: input_cpu.numpy()},
)[0]
onnx_logits = onnx_logits_array.astype(np.float32)

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
trt_logits = trt_logits.float()

print("测试图像：{}".format(image_path))
print("输入 shape/dtype：{} / {}".format(tuple(input_cpu.shape), input_cpu.dtype))
print("ONNX Runtime CPU 输出 shape：{}".format(onnx_logits.shape))
print("TensorRT 输出 shape：{}".format(tuple(trt_logits.shape)))

trt_logits = trt_logits.cpu().numpy().astype(np.float32)
absolute_error = np.abs(onnx_logits - trt_logits)
relative_error = absolute_error / np.maximum(np.abs(onnx_logits), 1e-8)
cosine_similarity = np.dot(onnx_logits.ravel(), trt_logits.ravel()) / (
    np.linalg.norm(onnx_logits.ravel()) * np.linalg.norm(trt_logits.ravel())
)

print("ONNX Runtime CPU vs TensorRT FP16")
print("MAE：{:.8g}".format(absolute_error.mean()))
print("最大绝对误差：{:.8g}".format(absolute_error.max()))
print("平均相对误差：{:.8g}".format(relative_error.mean()))
print("余弦相似度：{:.8f}".format(cosine_similarity))
