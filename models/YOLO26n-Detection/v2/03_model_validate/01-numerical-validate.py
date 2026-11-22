import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import tensorrt as trt
import torch
from ultralytics.data.augment import LetterBox

project_dir = Path(__file__).resolve().parent
model_dir = project_dir.parent
convert_dir = model_dir / "02_model_convert"
dataset_dir = model_dir / "01_model_train" / "datasets"
image_dir = dataset_dir / "holdout" / "images" / "val"
source_onnx = convert_dir / "onnx" / "yolo26n_fp32.onnx"
pruned_onnx = convert_dir / "onnx" / "yolo26n_pruned.onnx"
engine_path = convert_dir / "engine" / "yolo26n_pruned_int8.engine"
metrics_dir = project_dir / "metrics"
metrics_path = metrics_dir / "numerical.json"
image_size = 640
confidence_threshold = 0.001
match_iou_threshold = 0.5

for model_path in (source_onnx, pruned_onnx, engine_path):
    if not model_path.is_file():
        raise FileNotFoundError("缺少模型文件：{}；请先完成阶段 02".format(model_path))
image_paths = sorted(image_dir.glob("*.jpg"))
if not image_paths:
    raise FileNotFoundError("留出图片为空；先运行 ../01_model_train/01-prepare-calibration.py")

image_path = image_paths[0]
image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
if image is None:
    raise RuntimeError("无法读取验证图片：{}".format(image_path))
letterboxed = LetterBox(new_shape=(image_size, image_size), auto=False, scaleup=True)(image=image)
input_array = letterboxed[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
input_array = np.ascontiguousarray(input_array)

onnx_session = ort.InferenceSession(str(source_onnx), providers=["CPUExecutionProvider"])
pruned_session = ort.InferenceSession(str(pruned_onnx), providers=["CPUExecutionProvider"])
source_output = onnx_session.run(
    [onnx_session.get_outputs()[0].name],
    {onnx_session.get_inputs()[0].name: input_array},
)[0].astype(np.float32)
pruned_output = pruned_session.run(
    [pruned_session.get_outputs()[0].name],
    {pruned_session.get_inputs()[0].name: input_array},
)[0].astype(np.float32)

if not torch.cuda.is_available():
    raise RuntimeError("TensorRT INT8 engine 验证需要可用的 CUDA GPU")
runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
if engine is None:
    raise RuntimeError("TensorRT 无法加载 INT8 engine：{}".format(engine_path))
context = engine.create_execution_context()
if context is None:
    raise RuntimeError("TensorRT 无法创建 execution context")
input_names = [engine.get_tensor_name(index) for index in range(engine.num_io_tensors)
               if engine.get_tensor_mode(engine.get_tensor_name(index)) == trt.TensorIOMode.INPUT]
output_names = [engine.get_tensor_name(index) for index in range(engine.num_io_tensors)
                if engine.get_tensor_mode(engine.get_tensor_name(index)) == trt.TensorIOMode.OUTPUT]
if len(input_names) != 1 or len(output_names) != 1:
    raise RuntimeError("预期 INT8 engine 恰有一个输入和一个输出")
input_name, output_name = input_names[0], output_names[0]
if any(dimension < 0 for dimension in engine.get_tensor_shape(input_name)):
    if not context.set_input_shape(input_name, tuple(input_array.shape)):
        raise RuntimeError("TensorRT 无法设置输入 shape")
torch_types = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16}
if engine.get_tensor_dtype(input_name) not in torch_types or engine.get_tensor_dtype(output_name) not in torch_types:
    raise RuntimeError("当前 demo 预期 FP32/FP16 I/O，实际为 {} -> {}".format(
        engine.get_tensor_dtype(input_name), engine.get_tensor_dtype(output_name)
    ))
device_input = torch.from_numpy(input_array).to(device="cuda", dtype=torch_types[engine.get_tensor_dtype(input_name)]).contiguous()
device_output = torch.empty(
    tuple(context.get_tensor_shape(output_name)),
    dtype=torch_types[engine.get_tensor_dtype(output_name)],
    device="cuda",
)
stream = torch.cuda.Stream()
if not context.set_tensor_address(input_name, int(device_input.data_ptr())):
    raise RuntimeError("TensorRT 输入地址绑定失败")
if not context.set_tensor_address(output_name, int(device_output.data_ptr())):
    raise RuntimeError("TensorRT 输出地址绑定失败")
if not context.execute_async_v3(stream.cuda_stream):
    raise RuntimeError("TensorRT INT8 engine 执行失败")
stream.synchronize()
trt_output = device_output.float().cpu().numpy().astype(np.float32)

if source_output.shape != pruned_output.shape or pruned_output.shape != trt_output.shape:
    raise RuntimeError("模型输出 shape 不一致：source={} pruned={} TensorRT={}".format(
        source_output.shape, pruned_output.shape, trt_output.shape
    ))

def valid_rows(output):
    rows = output[0]
    return rows[(rows[:, 4] >= confidence_threshold) & (rows[:, 5] >= 0) & (rows[:, 5] < 80)]

def iou(left, right):
    intersection = max(0.0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0.0, min(left[3], right[3]) - max(left[1], right[1])
    )
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(left_area + right_area - intersection, 1.0e-12)

def compare_candidates(reference_output, converted_output):
    reference = valid_rows(reference_output)
    converted = valid_rows(converted_output)
    unmatched = set(range(len(converted)))
    matched = []
    for row in reference[np.argsort(-reference[:, 4], kind="stable")]:
        best_index = None
        best_iou = 0.0
        class_id = int(round(row[5]))
        for candidate_index in unmatched:
            candidate = converted[candidate_index]
            if int(round(candidate[5])) != class_id:
                continue
            candidate_iou = iou(row, candidate)
            if candidate_iou > best_iou:
                best_iou = candidate_iou
                best_index = candidate_index
        if best_index is not None and best_iou >= match_iou_threshold:
            candidate = converted[best_index]
            unmatched.remove(best_index)
            matched.append({
                "iou": best_iou,
                "box_coordinate_abs_errors": np.abs(row[:4] - candidate[:4]),
                "confidence_abs_error": abs(float(row[4] - candidate[4])),
            })
    box_errors = [error for item in matched for error in item["box_coordinate_abs_errors"]]
    return {
        "reference_valid_candidates": int(len(reference)),
        "converted_valid_candidates": int(len(converted)),
        "same_class_matches_iou_ge_0_5": len(matched),
        "unmatched_reference_candidates": int(len(reference) - len(matched)),
        "unmatched_converted_candidates": len(unmatched),
        "mean_matched_iou": float(np.mean([item["iou"] for item in matched])) if matched else None,
        "matched_box_coordinate_mae": float(np.mean(box_errors)) if box_errors else None,
        "matched_box_coordinate_max_abs_error": float(np.max(box_errors)) if box_errors else None,
        "matched_confidence_mae": float(np.mean([item["confidence_abs_error"] for item in matched])) if matched else None,
    }

def raw_tensor_summary(reference, converted):
    difference = np.abs(reference - converted)
    reference_flat = reference.ravel()
    converted_flat = converted.ravel()
    cosine = float(np.dot(reference_flat, converted_flat) / max(
        np.linalg.norm(reference_flat) * np.linalg.norm(converted_flat), 1.0e-12
    ))
    return {
        "mae_by_output_row": float(difference.mean()),
        "max_absolute_error_by_output_row": float(difference.max()),
        "cosine_similarity_by_output_row": cosine,
        "note": "模型输出按候选分数排序；剪枝/量化可能改变 TopK 顺序，此项逐行误差仅供参考，另报告同类 IoU 匹配误差。",
    }

metrics = {
    "image": str(image_path.relative_to(model_dir)),
    "input_shape": list(input_array.shape),
    "preprocessing": "Ultralytics LetterBox(640, auto=False, scaleup=True), BGR->RGB, NCHW, FP32 [0,1]",
    "runtime": {
        "onnxruntime": ort.__version__,
        "onnx_providers": onnx_session.get_providers(),
        "tensorrt": trt.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "engine_deserialize_and_execute": True,
        "engine_input": {"name": input_name, "shape": list(engine.get_tensor_shape(input_name)), "dtype": str(engine.get_tensor_dtype(input_name))},
        "engine_output": {"name": output_name, "shape": list(context.get_tensor_shape(output_name)), "dtype": str(engine.get_tensor_dtype(output_name))},
    },
    "outputs": {
        "fp32_onnx": {"shape": list(source_output.shape), "dtype": str(source_output.dtype)},
        "pruned_fp32_onnx": {"shape": list(pruned_output.shape), "dtype": str(pruned_output.dtype)},
        "tensorrt_int8": {"shape": list(trt_output.shape), "dtype": str(trt_output.dtype)},
    },
    "candidate_matching": {
        "confidence_threshold": confidence_threshold,
        "iou_threshold": match_iou_threshold,
        "fp32_onnx_vs_pruned_fp32_onnx": compare_candidates(source_output, pruned_output),
        "pruned_fp32_onnx_vs_tensorrt_int8": compare_candidates(pruned_output, trt_output),
    },
    "rowwise_tensor_comparison": {
        "fp32_onnx_vs_pruned_fp32_onnx": raw_tensor_summary(source_output, pruned_output),
        "pruned_fp32_onnx_vs_tensorrt_int8": raw_tensor_summary(pruned_output, trt_output),
    },
}
metrics_dir.mkdir(parents=True, exist_ok=True)
metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
print(json.dumps(metrics, indent=2))
print("数值验证结果：{}".format(metrics_path))
