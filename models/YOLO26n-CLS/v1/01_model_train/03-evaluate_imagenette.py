import json
from pathlib import Path
from time import perf_counter

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from ultralytics import YOLO
from ultralytics.data.augment import classify_transforms

project_dir = Path(__file__).resolve().parent
checkpoint_path = project_dir / "checkpoints" / "yolo26n-cls.pt"
validation_dir = project_dir / "datasets" / "imagenette2-small" / "val"
output_dir = project_dir / "metrics"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "imagenette_pytorch.json"

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

if not validation_dir.is_dir():
    raise FileNotFoundError("先运行 00-download_imagenette.sh：{}".format(validation_dir))

model = YOLO(str(checkpoint_path))
network = model.model.cuda().eval()
dataset = ImageFolder(str(validation_dir), transform=classify_transforms(224))
if set(dataset.classes) != set(imagenet_index):
    raise RuntimeError("Imagenette 类别目录与固定 ImageNet synset 映射不一致")
dataset.target_transform = lambda class_id: imagenet_index[dataset.classes[class_id]]
loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0, drop_last=False)

torch.cuda.reset_peak_memory_stats()
top1_correct = 0
top5_correct = 0
sample_count = 0
torch.cuda.synchronize()
started_at = perf_counter()
with torch.inference_mode():
    for images, targets in loader:
        logits = network(images.cuda(non_blocking=False))
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        if logits.ndim != 2 or logits.shape[1] != 1000:
            raise RuntimeError("模型输出应为 [batch, 1000]，实际为 {}".format(tuple(logits.shape)))
        top5 = logits.topk(5, dim=1).indices.cpu()
        targets = targets.cpu()
        top1_correct += int((top5[:, 0] == targets).sum())
        top5_correct += int((top5 == targets[:, None]).any(dim=1).sum())
        sample_count += targets.numel()
torch.cuda.synchronize()
elapsed_seconds = perf_counter() - started_at

metrics = {
    "model": "YOLO26n-cls",
    "checkpoint": "checkpoints/yolo26n-cls.pt",
    "dataset": "Imagenette2-160 deterministic 100-image subset",
    "samples": sample_count,
    "classes": len(dataset.classes),
    "imgsz": 224,
    "batch_size": 32,
    "top1_accuracy": top1_correct / sample_count,
    "top5_accuracy": top5_correct / sample_count,
    "elapsed_seconds": elapsed_seconds,
    "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 1024 ** 2,
}
output_path.write_text(json.dumps(metrics, indent=2) + "\n")
print(json.dumps(metrics, indent=2))
print("结果：{}".format(output_path))
