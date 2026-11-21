import json
import shutil
from pathlib import Path

import numpy as np
import onnxruntime as ort
import tensorrt as trt
import torch
from PIL import Image
from ultralytics.data.augment import classify_transforms

project_dir = Path(__file__).resolve().parent
convert_dir = project_dir.parent / "02_model_convert"
onnx_path = project_dir / "onnx" / "yolo26n-cls.onnx"
backbone_path = project_dir / "onnx" / "yolo26n-cls_backbone.onnx"
head_path = project_dir / "onnx" / "yolo26n-cls_head.onnx"
engine_path = project_dir / "engine" / "yolo26n-cls_fp16.engine"
image_path = project_dir / "sample.jpg"
metrics_dir = project_dir / "metrics"
metrics_dir.mkdir(parents=True, exist_ok=True)
output_path = metrics_dir / "numerical.json"

shutil.copy2(convert_dir / "onnx" / onnx_path.name, onnx_path)
shutil.copy2(convert_dir / "onnx" / backbone_path.name, backbone_path)
shutil.copy2(convert_dir / "onnx" / head_path.name, head_path)
shutil.copy2(convert_dir / "engine" / engine_path.name, engine_path)
input_tensor = classify_transforms(224)(Image.open(image_path).convert("RGB"))
input_array = input_tensor.unsqueeze(0).numpy().astype(np.float32)

session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
onnx_input_name = session.get_inputs()[0].name
onnx_output_name = session.get_outputs()[0].name
onnx_output = session.run([onnx_output_name], {onnx_input_name: input_array})[0].astype(np.float32)
backbone_session = ort.InferenceSession(str(backbone_path), providers=["CPUExecutionProvider"])
features = backbone_session.run(
    [backbone_session.get_outputs()[0].name],
    {backbone_session.get_inputs()[0].name: input_array},
)[0]
head_session = ort.InferenceSession(str(head_path), providers=["CPUExecutionProvider"])
split_output = head_session.run(
    [head_session.get_outputs()[0].name],
    {head_session.get_inputs()[0].name: features},
)[0].astype(np.float32)

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
    raise RuntimeError("预期一个输入和一个输出 tensor")

input_name = input_names[0]
output_name = output_names[0]
torch_dtypes = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16}
input_dtype = torch_dtypes[engine.get_tensor_dtype(input_name)]
output_dtype = torch_dtypes[engine.get_tensor_dtype(output_name)]
if any(dimension < 0 for dimension in engine.get_tensor_shape(input_name)):
    if not context.set_input_shape(input_name, tuple(input_array.shape)):
        raise RuntimeError("TensorRT 无法设置输入 shape")
trt_input = torch.from_numpy(input_array).to(device="cuda", dtype=input_dtype).contiguous()
trt_output = torch.empty(tuple(context.get_tensor_shape(output_name)), dtype=output_dtype, device="cuda")
stream = torch.cuda.Stream()
if not context.set_tensor_address(input_name, int(trt_input.data_ptr())):
    raise RuntimeError("TensorRT 输入地址绑定失败")
if not context.set_tensor_address(output_name, int(trt_output.data_ptr())):
    raise RuntimeError("TensorRT 输出地址绑定失败")
if not context.execute_async_v3(stream.cuda_stream):
    raise RuntimeError("TensorRT inference 执行失败")
stream.synchronize()
trt_array = trt_output.float().cpu().numpy().astype(np.float32)

if onnx_output.shape != trt_array.shape:
    raise RuntimeError("输出形状不同：ONNX {}，TensorRT {}".format(onnx_output.shape, trt_array.shape))
absolute_error = np.abs(onnx_output - trt_array)
relative_error = absolute_error / np.maximum(np.abs(onnx_output), 1e-8)
onnx_flat = onnx_output.ravel()
trt_flat = trt_array.ravel()
cosine_similarity = float(np.dot(onnx_flat, trt_flat) / (np.linalg.norm(onnx_flat) * np.linalg.norm(trt_flat)))
split_error = np.abs(onnx_output - split_output)
metrics = {
    "image": str(image_path.relative_to(project_dir)),
    "input_shape": list(input_array.shape),
    "output_shape": list(onnx_output.shape),
    "onnx_cpu_vs_tensorrt_fp16": {
        "mae": float(absolute_error.mean()),
        "max_absolute_error": float(absolute_error.max()),
        "mean_relative_error": float(relative_error.mean()),
        "cosine_similarity": cosine_similarity,
        "top1_agreement": int(onnx_output[0].argmax()) == int(trt_array[0].argmax()),
    },
    "onnx_full_vs_backbone_plus_head": {
        "mae": float(split_error.mean()),
        "max_absolute_error": float(split_error.max()),
        "top1_agreement": int(onnx_output[0].argmax()) == int(split_output[0].argmax()),
    },
}
output_path.write_text(json.dumps(metrics, indent=2) + "\n")
print(json.dumps(metrics, indent=2))
print("结果：{}".format(output_path))
