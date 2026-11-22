import json
from pathlib import Path
from time import perf_counter

import torch
from ultralytics import YOLO
from ultralytics.utils import checks

checks.AUTOINSTALL = False

project_dir = Path(__file__).resolve().parent
model_dir = project_dir.parent
train_dir = model_dir / "01_model_train"
convert_dir = model_dir / "02_model_convert"
data_yaml = train_dir / "datasets" / "holdout" / "data.yaml"
models = {
    "pytorch_fp32_original": train_dir / "checkpoints" / "yolo26n.pt",
    "onnx_fp32_original": convert_dir / "onnx" / "yolo26n_fp32.onnx",
    "onnx_fp32_pruned": convert_dir / "onnx" / "yolo26n_pruned.onnx",
    "tensorrt_int8_pruned": convert_dir / "engine" / "yolo26n_pruned_int8.engine",
}
metrics_dir = project_dir / "metrics"
output_path = metrics_dir / "task_accuracy.json"
expected_image_count = 8

if not data_yaml.is_file():
    raise FileNotFoundError("先运行 ../01_model_train/01-prepare-calibration.py：{}".format(data_yaml))
image_dir = train_dir / "datasets" / "holdout" / "images" / "val"
actual_image_count = len(list(image_dir.glob("*.jpg")))
if actual_image_count != expected_image_count:
    raise RuntimeError("预期留出集 {} 张图片，实际为 {} 张".format(expected_image_count, actual_image_count))
for model_path in models.values():
    if not model_path.is_file():
        raise FileNotFoundError("缺少模型文件：{}；请先完成阶段 02".format(model_path))

if not torch.cuda.is_available():
    raise RuntimeError("此验证流程要求在 Jetson CUDA 环境运行")
metrics_dir.mkdir(parents=True, exist_ok=True)
results = {}
for name, model_path in models.items():
    device = "cpu" if name.startswith("onnx_") else 0
    print("\n=== {} ({}) ===".format(name, device), flush=True)
    if device == 0:
        torch.cuda.reset_peak_memory_stats()
    detector = YOLO(str(model_path), task="detect")
    started_at = perf_counter()
    validation = detector.val(
        data=str(data_yaml),
        imgsz=640,
        batch=1,
        device=device,
        workers=0,
        plots=False,
        verbose=False,
        project=str(metrics_dir / "ultralytics_runs"),
        name=name,
        exist_ok=True,
    )
    if device == 0:
        torch.cuda.synchronize()
    results[name] = {
        "model": str(model_path.relative_to(model_dir)),
        "device": torch.cuda.get_device_name(0) if device == 0 else "CPU",
        "dataset_images": actual_image_count,
        "imgsz": 640,
        "batch_size": 1,
        "metrics": {
            "precision": float(validation.box.mp),
            "recall": float(validation.box.mr),
            "map50": float(validation.box.map50),
            "map50_95": float(validation.box.map),
        },
        "speed_ms_per_image": {key: float(value) for key, value in validation.speed.items()},
        "validation_elapsed_seconds": perf_counter() - started_at,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 1024 ** 2 if device == 0 else None,
        "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / 1024 ** 2 if device == 0 else None,
    }
    print(json.dumps(results[name], indent=2), flush=True)

reference = results["pytorch_fp32_original"]["metrics"]
for row in results.values():
    row["delta_vs_pytorch_fp32"] = {
        key: row["metrics"][key] - reference[key]
        for key in ("precision", "recall", "map50", "map50_95")
    }

metrics = {
    "dataset": "COCO val2017 deterministic 8-image held-out subset; smoke/regression only, not a full COCO benchmark",
    "calibration_images_excluded": True,
    "preprocessing": "Ultralytics LetterBox, imgsz=640",
    "postprocessing": "Ultralytics detection validator; same validator settings for all backends",
    "backend_devices": {
        "PyTorch and TensorRT": "CUDA GPU",
        "ONNX Runtime": "CPUExecutionProvider",
        "note": "Task metrics are comparable; per-image speed includes different backend/device settings and is not a controlled latency benchmark.",
    },
    "acceptance_note": "Small holdout metrics are for pipeline regression only; they cannot establish full-dataset accuracy acceptance.",
    "models": results,
}
output_path.write_text(json.dumps(metrics, indent=2) + "\n")
print("\n任务精度结果：{}".format(output_path))
