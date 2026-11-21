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
train_dir = project_dir.parent / "01_model_train"
onnx_path = project_dir / "onnx" / "yolo26n-cls.onnx"
engine_path = project_dir / "engine" / "yolo26n-cls_fp16.engine"
validation_dir = train_dir / "datasets" / "imagenette2-small" / "val"
metrics_dir = project_dir / "metrics"
metrics_dir.mkdir(parents=True, exist_ok=True)
output_path = metrics_dir / "task_accuracy.json"

shutil.copy2(convert_dir / "onnx" / onnx_path.name, onnx_path)
shutil.copy2(convert_dir / "engine" / engine_path.name, engine_path)
if not validation_dir.is_dir():
    raise FileNotFoundError("先运行 01_model_train/00-download_imagenette.sh")

# ImageNet synset folder -> ImageNet-1K class index.
imagenet_index = {
    "n01440764": 0,
    "n02102040": 217,
    "n02979186": 482,
    "n03000684": 491,
    "n03028079": 497,
    "n03394916": 566,
    "n03417042": 569,
    "n03425413": 571,
    "n03445777": 574,
    "n03888257": 701,
}
image_paths = sorted(
    path for path in validation_dir.rglob("*")
    if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
)
if len(image_paths) != 100:
    raise RuntimeError("预期 100 张验证图，实际找到 {} 张".format(len(image_paths)))

session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
onnx_input_name = session.get_inputs()[0].name
onnx_output_name = session.get_outputs()[0].name
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
if any(dimension < 0 for dimension in engine.get_tensor_shape(input_name)):
    if not context.set_input_shape(input_name, (1, 3, 224, 224)):
        raise RuntimeError("TensorRT 无法设置输入 shape")
torch_dtypes = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16}
input_dtype = torch_dtypes[engine.get_tensor_dtype(input_name)]
output_dtype = torch_dtypes[engine.get_tensor_dtype(output_name)]
trt_input = torch.empty(tuple(context.get_tensor_shape(input_name)), dtype=input_dtype, device="cuda")
trt_output = torch.empty(tuple(context.get_tensor_shape(output_name)), dtype=output_dtype, device="cuda")
stream = torch.cuda.Stream()
if not context.set_tensor_address(input_name, int(trt_input.data_ptr())):
    raise RuntimeError("TensorRT 输入地址绑定失败")
if not context.set_tensor_address(output_name, int(trt_output.data_ptr())):
    raise RuntimeError("TensorRT 输出地址绑定失败")
transform = classify_transforms(224)

onnx_top1 = 0
onnx_top5 = 0
trt_top1 = 0
trt_top5 = 0
for sample_index, image_path in enumerate(image_paths, start=1):
    synset = image_path.parent.name
    if synset not in imagenet_index:
        raise RuntimeError("未知 ImageNet synset：{}".format(synset))
    target = imagenet_index[synset]
    input_array = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).numpy().astype(np.float32)

    onnx_output = session.run(
        [onnx_output_name],
        {onnx_input_name: input_array},
    )[0][0]
    onnx_indices = np.argsort(onnx_output)[-5:]
    onnx_top1 += int(int(onnx_indices[-1]) == target)
    onnx_top5 += int(target in onnx_indices)

    trt_input.copy_(torch.from_numpy(input_array).to(device="cuda", dtype=input_dtype))
    if not context.execute_async_v3(stream.cuda_stream):
        raise RuntimeError("TensorRT inference 失败：{}".format(image_path))
    stream.synchronize()
    trt_values = trt_output.float()[0].cpu().numpy()
    trt_indices = np.argsort(trt_values)[-5:]
    trt_top1 += int(int(trt_indices[-1]) == target)
    trt_top5 += int(target in trt_indices)
    if sample_index % 500 == 0:
        print("已验证 {}/{} 张".format(sample_index, len(image_paths)), flush=True)

sample_count = len(image_paths)
train_metrics_path = train_dir / "metrics" / "imagenette_pytorch.json"
if not train_metrics_path.is_file():
    raise FileNotFoundError("先运行 01_model_train/03-evaluate_imagenette.py")
train_metrics = json.loads(train_metrics_path.read_text())
metrics = {
    "dataset": "Imagenette2-160 deterministic 100-image subset",
    "samples": sample_count,
    "classes": 10,
    "imgsz": 224,
    "preprocess": "Ultralytics classify_transforms(224)",
    "pytorch_reference": {
        "top1_accuracy": train_metrics["top1_accuracy"],
        "top5_accuracy": train_metrics["top5_accuracy"],
    },
    "onnxruntime_cpu": {
        "top1_accuracy": onnx_top1 / sample_count,
        "top5_accuracy": onnx_top5 / sample_count,
    },
    "tensorrt_fp16": {
        "top1_accuracy": trt_top1 / sample_count,
        "top5_accuracy": trt_top5 / sample_count,
    },
}
metrics["delta_vs_pytorch_percentage_points"] = {
    "onnx_top1": 100.0 * (metrics["onnxruntime_cpu"]["top1_accuracy"] - train_metrics["top1_accuracy"]),
    "onnx_top5": 100.0 * (metrics["onnxruntime_cpu"]["top5_accuracy"] - train_metrics["top5_accuracy"]),
    "tensorrt_top1": 100.0 * (metrics["tensorrt_fp16"]["top1_accuracy"] - train_metrics["top1_accuracy"]),
    "tensorrt_top5": 100.0 * (metrics["tensorrt_fp16"]["top5_accuracy"] - train_metrics["top5_accuracy"]),
}
output_path.write_text(json.dumps(metrics, indent=2) + "\n")
print(json.dumps(metrics, indent=2))
print("结果：{}".format(output_path))
