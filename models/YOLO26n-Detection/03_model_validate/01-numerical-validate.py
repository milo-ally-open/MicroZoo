import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import tensorrt as trt
import torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils import checks

checks.AUTOINSTALL = False

project_dir = Path(__file__).resolve().parent
model_dir = project_dir.parent
convert_dir = model_dir / "02_model_convert"
train_dir = model_dir / "01_model_train"
onnx_dir = project_dir / "onnx"
engine_dir = project_dir / "engine"
onnx_dir.mkdir(parents=True, exist_ok=True)
engine_dir.mkdir(parents=True, exist_ok=True)
onnx_path = onnx_dir / "yolo26n.onnx"
engine_path = engine_dir / "yolo26n_fp16.engine"
checkpoint_path = train_dir / "checkpoints" / "yolo26n.pt"
selection_path = train_dir / "datasets" / "coco-val-32" / "selection.txt"
metrics_dir = project_dir / "metrics"
metrics_dir.mkdir(parents=True, exist_ok=True)
output_path = metrics_dir / "numerical.json"

shutil.copy2(convert_dir / "onnx" / onnx_path.name, onnx_path)
shutil.copy2(convert_dir / "engine" / engine_path.name, engine_path)
if not selection_path.is_file():
    raise FileNotFoundError("先运行 01_model_train/00-prepare_coco_val.py")
image_name = next(line.split()[1] for line in selection_path.read_text().splitlines() if line and not line.startswith("#"))
source_image = train_dir / "datasets" / "coco-val-32" / "images" / "val" / image_name
sample_path = project_dir / "sample.jpg"
if not sample_path.is_file():
    shutil.copy2(source_image, sample_path)

image = cv2.imread(str(sample_path), cv2.IMREAD_COLOR)
if image is None:
    raise RuntimeError("无法读取验证图片：{}".format(sample_path))
letterboxed = LetterBox(new_shape=(640, 640), auto=False, scaleup=True)(image=image)
input_array = letterboxed[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
input_array = np.ascontiguousarray(input_array)

backend_paths = {
    "pytorch_fp32": train_dir / "checkpoints" / "yolo26n.pt",
    "onnxruntime_cpu": onnx_path,
    "tensorrt_fp16": engine_path,
}
backend_devices = {"pytorch_fp32": 0 if torch.cuda.is_available() else "cpu", "onnxruntime_cpu": "cpu", "tensorrt_fp16": 0}
detections = {}
for backend_name, model_path in backend_paths.items():
    print("运行 {} ...".format(backend_name), flush=True)
    backend = YOLO(str(model_path), task="detect")
    result = backend.predict(
        source=str(sample_path),
        imgsz=640,
        conf=0.001,
        iou=0.7,
        max_det=300,
        device=backend_devices[backend_name],
        verbose=False,
    )[0]
    if backend_devices[backend_name] == 0:
        torch.cuda.synchronize()
    detections[backend_name] = []
    for index in range(len(result.boxes)):
        detections[backend_name].append({
            "class_id": int(result.boxes.cls[index].item()),
            "confidence": float(result.boxes.conf[index].item()),
            "xyxy": [float(value) for value in result.boxes.xyxy[index].tolist()],
        })

ort_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
ort_input_name = ort_session.get_inputs()[0].name
ort_output_name = ort_session.get_outputs()[0].name
onnx_tensor = ort_session.run([ort_output_name], {ort_input_name: input_array})[0].astype(np.float32)

trt_runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
engine = trt_runtime.deserialize_cuda_engine(engine_path.read_bytes())
if engine is None:
    raise RuntimeError("TensorRT 无法反序列化 engine")
context = engine.create_execution_context()
input_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)
               if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT]
output_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)
                if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT]
if len(input_names) != 1 or len(output_names) != 1:
    raise RuntimeError("当前脚本要求 engine 恰有一个输入和一个输出")
input_name, output_name = input_names[0], output_names[0]
trt_dtypes = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16}
input_dtype = trt_dtypes[engine.get_tensor_dtype(input_name)]
output_dtype = trt_dtypes[engine.get_tensor_dtype(output_name)]
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
    raise RuntimeError("TensorRT TensorRT engine 执行失败")
stream.synchronize()
trt_tensor = trt_output.float().cpu().numpy().astype(np.float32)
if onnx_tensor.shape != trt_tensor.shape:
    raise RuntimeError("ONNX 与 TensorRT 原始输出 shape 不同：{} vs {}".format(onnx_tensor.shape, trt_tensor.shape))

onnx_rows = onnx_tensor[0]
trt_rows = trt_tensor[0]
onnx_valid = onnx_rows[(onnx_rows[:, 4] >= 0.001) & (onnx_rows[:, 5] >= 0) & (onnx_rows[:, 5] < 80)]
trt_valid = trt_rows[(trt_rows[:, 4] >= 0.001) & (trt_rows[:, 5] >= 0) & (trt_rows[:, 5] < 80)]
tensor_comparison = {"onnx_cpu_vs_tensorrt_fp16": {
    "raw_output_shape": list(onnx_tensor.shape),
    "confidence_threshold_for_comparison": 0.001,
    "onnx_valid_candidates": int(len(onnx_valid)),
    "tensorrt_valid_candidates": int(len(trt_valid)),
}}
if len(onnx_valid) == len(trt_valid) and len(onnx_valid) > 0:
    tensor_error = np.abs(onnx_valid - trt_valid)
    tensor_reference = onnx_valid.ravel()
    tensor_converted = trt_valid.ravel()
    tensor_comparison["onnx_cpu_vs_tensorrt_fp16"].update({
        "class_agreement": bool(np.array_equal(onnx_valid[:, 5].astype(np.int32), trt_valid[:, 5].astype(np.int32))),
        "mae_valid_rows": float(tensor_error.mean()),
        "max_absolute_error_valid_rows": float(tensor_error.max()),
        "box_coordinate_mae": float(np.abs(onnx_valid[:, :4] - trt_valid[:, :4]).mean()),
        "box_coordinate_max_absolute_error": float(np.abs(onnx_valid[:, :4] - trt_valid[:, :4]).max()),
        "confidence_mae": float(np.abs(onnx_valid[:, 4] - trt_valid[:, 4]).mean()),
        "cosine_similarity_valid_rows": float(np.dot(tensor_reference, tensor_converted) / max(
            np.linalg.norm(tensor_reference) * np.linalg.norm(tensor_converted), 1e-12
        )),
    })
else:
    tensor_comparison["onnx_cpu_vs_tensorrt_fp16"].update({
        "class_agreement": False,
        "note": "ONNX 与 TensorRT 的高置信候选数量不同，不能逐行计算数值误差；请结合框匹配与 task mAP。",
    })

comparison_rows = {}
reference_detections = detections["pytorch_fp32"]
for backend_name in ["onnxruntime_cpu", "tensorrt_fp16"]:
    candidates = detections[backend_name]
    unmatched = set(range(len(candidates)))
    matched_rows = []
    for reference in reference_detections:
        best_index = None
        best_iou = 0.0
        for candidate_index in unmatched:
            candidate = candidates[candidate_index]
            if candidate["class_id"] != reference["class_id"]:
                continue
            ax1, ay1, ax2, ay2 = reference["xyxy"]
            bx1, by1, bx2, by2 = candidate["xyxy"]
            intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
            area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
            area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
            iou = intersection / max(area_a + area_b - intersection, 1e-12)
            if iou > best_iou:
                best_iou = iou
                best_index = candidate_index
        if best_index is not None and best_iou >= 0.5:
            candidate = candidates[best_index]
            unmatched.remove(best_index)
            matched_rows.append({
                "iou": best_iou,
                "max_box_coordinate_abs_error": max(abs(a - b) for a, b in zip(reference["xyxy"], candidate["xyxy"])),
                "confidence_abs_error": abs(reference["confidence"] - candidate["confidence"]),
            })
    comparison_rows["pytorch_fp32_vs_{}".format(backend_name)] = {
        "pytorch_detections": len(reference_detections),
        "converted_detections": len(candidates),
        "matched_same_class_iou_ge_0_5": len(matched_rows),
        "unmatched_pytorch_detections": len(reference_detections) - len(matched_rows),
        "unmatched_converted_detections": len(unmatched),
        "mean_matched_iou": float(np.mean([row["iou"] for row in matched_rows])) if matched_rows else None,
        "max_box_coordinate_abs_error": max((row["max_box_coordinate_abs_error"] for row in matched_rows), default=None),
        "mean_confidence_abs_error": float(np.mean([row["confidence_abs_error"] for row in matched_rows])) if matched_rows else None,
    }

metrics = {
    "image": str(sample_path.relative_to(project_dir)),
    "input_shape": list(input_array.shape),
    "preprocessing": "same Ultralytics LetterBox 640, RGB, NCHW, [0,1]",
    "onnx_output": {"name": ort_output_name, "shape": list(onnx_tensor.shape), "type": ort_session.get_outputs()[0].type},
    "tensorrt_output": {"name": output_name, "shape": list(trt_tensor.shape), "dtype": str(engine.get_tensor_dtype(output_name))},
    "runtime_compatibility": {
        "onnxruntime": ort.__version__,
        "onnx_providers": ort_session.get_providers(),
        "tensorrt": trt.__version__,
        "engine_deserialize_and_execute": True,
    },
    "raw_tensor_comparison": tensor_comparison,
    "postprocessed_detection_comparison": comparison_rows,
    "detections": detections,
}
output_path.write_text(json.dumps(metrics, indent=2) + "\n")
print(json.dumps(metrics, indent=2))
print("结果：{}".format(output_path))
