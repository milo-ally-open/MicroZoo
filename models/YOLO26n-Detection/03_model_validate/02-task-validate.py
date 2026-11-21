import json
import shutil
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
onnx_path = project_dir / "onnx" / "yolo26n.onnx"
engine_path = project_dir / "engine" / "yolo26n_fp16.engine"
data_yaml = train_dir / "datasets" / "coco-val-32" / "data.yaml"
metrics_dir = project_dir / "metrics"
metrics_dir.mkdir(parents=True, exist_ok=True)
output_path = metrics_dir / "task_accuracy.json"

shutil.copy2(convert_dir / "onnx" / onnx_path.name, onnx_path)
shutil.copy2(convert_dir / "engine" / engine_path.name, engine_path)
if not data_yaml.is_file():
    raise FileNotFoundError("先运行 01_model_train/00-prepare_coco_val.py")

model_paths = {
    "pytorch_fp32": train_dir / "checkpoints" / "yolo26n.pt",
    "onnxruntime_cpu": onnx_path,
    "tensorrt_fp16": engine_path,
}
devices = {"pytorch_fp32": 0 if torch.cuda.is_available() else "cpu", "onnxruntime_cpu": "cpu", "tensorrt_fp16": 0}
results_by_model = {}
for name, model_path in model_paths.items():
    print("\n=== {} ===".format(name), flush=True)
    if not model_path.is_file():
        raise FileNotFoundError("缺少模型文件：{}".format(model_path))
    model = YOLO(str(model_path), task="detect")
    if devices[name] == 0:
        torch.cuda.reset_peak_memory_stats()
    started_at = perf_counter()
    validation = model.val(
        data=str(data_yaml),
        imgsz=640,
        batch=1,
        device=devices[name],
        workers=0,
        plots=False,
        verbose=False,
        project=str(metrics_dir / "ultralytics_runs"),
        name=name,
        exist_ok=True,
    )
    if devices[name] == 0:
        torch.cuda.synchronize()
    elapsed_seconds = perf_counter() - started_at
    results_by_model[name] = {
        "model_path": str(model_path.relative_to(model_dir)),
        "device": torch.cuda.get_device_name(0) if devices[name] == 0 else "CPU",
        "dataset_images": 32,
        "imgsz": 640,
        "batch_size": 1,
        "metrics": {
            "precision": float(validation.box.mp),
            "recall": float(validation.box.mr),
            "map50": float(validation.box.map50),
            "map50_95": float(validation.box.map),
        },
        "speed_ms_per_image": {key: float(value) for key, value in validation.speed.items()},
        "validation_elapsed_seconds": elapsed_seconds,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 1024 ** 2 if devices[name] == 0 else None,
        "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / 1024 ** 2 if devices[name] == 0 else None,
    }
    print(json.dumps(results_by_model[name], indent=2), flush=True)

metrics = {
    "dataset": "COCO 2017 val2017 deterministic 32-image subset; smoke/regression only, not full COCO benchmark",
    "postprocessing": "Ultralytics detection validator, confidence and IoU defaults pinned by ultralytics==8.4.155",
    "models": results_by_model,
}
output_path.write_text(json.dumps(metrics, indent=2) + "\n")
print("\n结果：{}".format(output_path))
