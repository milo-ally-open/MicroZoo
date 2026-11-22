import json
import resource
from pathlib import Path
from time import perf_counter

import torch
from ultralytics import YOLO

project_dir = Path(__file__).resolve().parent
checkpoint_path = project_dir / "checkpoints" / "yolo26n.pt"
dataset_dir = project_dir / "datasets" / "coco-val-32"
selection_path = dataset_dir / "selection.txt"
metrics_dir = project_dir / "metrics"
metrics_dir.mkdir(parents=True, exist_ok=True)
output_path = metrics_dir / "single_image.json"

if not checkpoint_path.is_file():
    raise FileNotFoundError("先运行 01-download_weights.py")
if not selection_path.is_file():
    raise FileNotFoundError("先运行 00-prepare_coco_val.py")
image_name = selection_path.read_text().splitlines()[2].split()[1]
image_path = dataset_dir / "images" / "val" / image_name
device = 0 if torch.cuda.is_available() else "cpu"

model = YOLO(str(checkpoint_path))
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

for _ in range(3):
    model.predict(source=str(image_path), imgsz=640, device=device, verbose=False)
if torch.cuda.is_available():
    torch.cuda.synchronize()
started_at = perf_counter()
result = None
for _ in range(20):
    result = model.predict(source=str(image_path), imgsz=640, device=device, verbose=False)[0]
if torch.cuda.is_available():
    torch.cuda.synchronize()
inference_ms = (perf_counter() - started_at) * 1000.0 / 20

boxes = result.boxes
predictions = []
for index in range(len(boxes)):
    class_id = int(boxes.cls[index].item())
    predictions.append({
        "class_id": class_id,
        "class_name": model.names[class_id],
        "confidence": float(boxes.conf[index].item()),
        "xyxy": [float(value) for value in boxes.xyxy[index].tolist()],
    })

metrics = {
    "model": "YOLO26n COCO pretrained; no project fine-tuning",
    "checkpoint": "checkpoints/yolo26n.pt",
    "image": str(image_path.relative_to(project_dir)),
    "input_size": [1, 3, 640, 640],
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    "parameter_count": sum(parameter.numel() for parameter in model.model.parameters()),
    "checkpoint_mib": checkpoint_path.stat().st_size / 1024 ** 2,
    "inference_ms_20_run_average": inference_ms,
    "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else None,
    "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / 1024 ** 2 if torch.cuda.is_available() else None,
    "max_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    "detection_count": len(predictions),
    "detections": predictions,
}
output_path.write_text(json.dumps(metrics, indent=2) + "\n")
print(json.dumps(metrics, indent=2))
print("结果：{}".format(output_path))
