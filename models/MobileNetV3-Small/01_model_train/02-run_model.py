from pathlib import Path
from time import perf_counter

import torch
from PIL import Image
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

project_dir = Path(__file__).resolve().parent
checkpoint_path = project_dir / "checkpoints" / "mobilenet_v3_small-047dcff4.pth"
image_path = project_dir / "dog.jpg"
weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
device = torch.device("cuda")

model = mobilenet_v3_small(weights=None)
model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
model = model.to(device).eval()

input_tensor = weights.transforms()(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

torch.cuda.reset_peak_memory_stats(device)
torch.cuda.synchronize(device)
started_at = perf_counter()
with torch.inference_mode():
    logits = model(input_tensor)
torch.cuda.synchronize(device)
inference_seconds = perf_counter() - started_at

probabilities = logits[0].softmax(dim=0)
top_scores, top_indices = probabilities.topk(5)
categories = weights.meta["categories"]

print("测试图像：{}".format(image_path))
print("执行设备：{}".format(torch.cuda.get_device_name(device)))
print("模型参数量：{:,}".format(sum(parameter.numel() for parameter in model.parameters())))
print("参数显存：{:.2f} MiB".format(sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()) / 1024 ** 2))
print("单次推理耗时：{:.2f} ms".format(inference_seconds * 1000))
print("GPU 峰值已分配显存：{:.2f} MiB".format(torch.cuda.max_memory_allocated(device) / 1024 ** 2))
print("Top-5 预测：")
for score, index in zip(top_scores.tolist(), top_indices.tolist()):
    print("{}: {:.2%}".format(categories[index], score))
print("单张样例仅用于确认模型加载与推理；分类准确率和 F1 仍需独立标注测试集验收。")
