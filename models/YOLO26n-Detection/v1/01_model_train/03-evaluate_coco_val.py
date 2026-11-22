import json
from pathlib import Path

import torch
from ultralytics import YOLO

project_dir = Path(__file__).resolve().parent
checkpoint_path = project_dir / "checkpoints" / "yolo26n.pt"
dataset_dir = project_dir / "datasets" / "coco-val-32"
data_yaml = dataset_dir / "data.yaml"
metrics_dir = project_dir / "metrics"
metrics_dir.mkdir(parents=True, exist_ok=True)
output_path = metrics_dir / "coco_val_32_pytorch.json"

if not checkpoint_path.is_file():
    raise FileNotFoundError("先运行 01-download_weights.py")
if not data_yaml.is_file():
    raise FileNotFoundError("先运行 00-prepare_coco_val.py")

model = YOLO(str(checkpoint_path))
parameter_count = sum(parameter.numel() for parameter in model.model.parameters())
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
results = model.val(
    data=str(data_yaml),
    imgsz=640,
    batch=1,
    device=0 if torch.cuda.is_available() else "cpu",
    workers=0,
    plots=False,
    verbose=False,
    project=str(metrics_dir / "runs"),
    name="pytorch_coco_val_32",
    exist_ok=True,
)
if torch.cuda.is_available():
    torch.cuda.synchronize()

metrics = {
    "model": "YOLO26n COCO pretrained; no project fine-tuning",
    "checkpoint": "checkpoints/yolo26n.pt",
    "dataset": "COCO 2017 val2017 fixed 32-image subset",
    "dataset_yaml": str(data_yaml.relative_to(project_dir)),
    "imgsz": 640,
    "batch_size": 1,
    "parameter_count": parameter_count,
    "fp32_weight_estimate_mib": parameter_count * 4 / 1024 ** 2,
    "fp16_weight_estimate_mib": parameter_count * 2 / 1024 ** 2,
    "checkpoint_mib": checkpoint_path.stat().st_size / 1024 ** 2,
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    "precision": "FP32 PyTorch",
    "images": sum(1 for line in (dataset_dir / "selection.txt").read_text().splitlines() if line and not line.startswith("#")),
    "metrics": {
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "map50": float(results.box.map50),
        "map50_95": float(results.box.map),
    },
    "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else None,
    "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / 1024 ** 2 if torch.cuda.is_available() else None,
}
output_path.write_text(json.dumps(metrics, indent=2) + "\n")
print(json.dumps(metrics, indent=2))
print("结果：{}".format(output_path))
