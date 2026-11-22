from pathlib import Path

import torch
from PIL import Image
from ultralytics import YOLO
from ultralytics.data.augment import classify_transforms

project_dir = Path(__file__).resolve().parent
checkpoint_path = project_dir / "pt" / "yolo26n-cls.pt"
image_path = project_dir.parent / "01_model_train" / "sample.jpg"
model = YOLO(str(checkpoint_path))
network = model.model.eval()
input_tensor = classify_transforms(224)(Image.open(image_path).convert("RGB")).unsqueeze(0)

print("模型：YOLO26n-cls，权重已加载：{}".format(checkpoint_path))
print("输入：{} {}".format(tuple(input_tensor.shape), input_tensor.dtype))
with torch.inference_mode():
    for index, layer in enumerate(network.model):
        input_tensor = layer(input_tensor)
        if isinstance(input_tensor, (tuple, list)):
            input_tensor = input_tensor[0]
        print(
            "model[{}] {:<24s} -> {} {} ({:.2f} MiB)".format(
                index,
                layer.__class__.__name__,
                tuple(input_tensor.shape),
                input_tensor.dtype,
                input_tensor.numel() * input_tensor.element_size() / 1024 ** 2,
            )
        )
