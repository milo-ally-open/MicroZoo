from pathlib import Path
from time import perf_counter

import torch
from ultralytics import YOLO

project_dir = Path(__file__).resolve().parent
checkpoint_path = project_dir / "checkpoints" / "yolo26n-cls.pt"
image_path = project_dir / "sample.jpg"
device = 0 if torch.cuda.is_available() else "cpu"

model = YOLO(str(checkpoint_path))
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

started_at = perf_counter()
result = model.predict(
    source=str(image_path),
    imgsz=224,
    device=device,
    verbose=False,
)[0]
if torch.cuda.is_available():
    torch.cuda.synchronize()
inference_ms = (perf_counter() - started_at) * 1000.0

probabilities = result.probs.data
top_scores, top_indices = probabilities.topk(5)
print("模型：YOLO26n-cls，输入尺寸：224 × 224")
print("设备：{}".format(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"))
print("图片：{}".format(image_path))
print("参数量：{:,}".format(sum(parameter.numel() for parameter in model.model.parameters())))
print("权重文件：{:.2f} MiB".format(checkpoint_path.stat().st_size / 1024 ** 2))
print("单图 predict 耗时（含预处理）：{:.2f} ms".format(inference_ms))
if torch.cuda.is_available():
    print("PyTorch CUDA 峰值分配：{:.2f} MiB".format(torch.cuda.max_memory_allocated() / 1024 ** 2))
print("Top-5：")
for score, index in zip(top_scores.tolist(), top_indices.tolist()):
    print("  {} ({}, {:.2%})".format(model.names[index], index, score))
