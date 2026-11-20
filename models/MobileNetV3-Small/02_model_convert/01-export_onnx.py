from pathlib import Path

import torch
from torchvision.models import mobilenet_v3_small

project_dir = Path(__file__).resolve().parent
checkpoint_path = project_dir / "pt" / "mobilenet_v3_small-047dcff4.pt"
output_path = project_dir / "onnx" / "mobilenet_v3_small.onnx"

output_path.parent.mkdir(parents=True, exist_ok=True)

model = mobilenet_v3_small(weights=None)
model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
model.eval()

input_tensor = torch.zeros(1, 3, 224, 224, dtype=torch.float32)

torch.onnx.export(
    model,
    input_tensor,
    output_path,
    input_names=["images"],
    output_names=["logits"],
    opset_version=17,
    do_constant_folding=True,
    dynamo=False,
)

print("ONNX 模型：{}".format(output_path))
print("输入：images，FP32，NCHW，形状 [1, 3, 224, 224]，RGB 归一化图像")
print("输出：logits，FP32，形状 [1, 1000]，ImageNet-1K 类别分数")
