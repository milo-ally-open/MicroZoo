"""将 MobileNetV3-Small 切分为 backbone 和分类头。

切分的目的，是把 通用特征提取 和 具体任务的预测 分开

1. 运行 `/usr/bin/python3 models/MobileNetV3-Small/02_model_convert/02-check_layers.py`。
   检查 `features[12]` 的输出应为 `[1, 576, 7, 7]`；之后的 `avgpool`、
   `flatten` 和 `classifier` 构成分类头。
2. 在 https://netron.app 打开 `onnx/mobilenet_v3_small.onnx`。找到全局平均池化
   节点（通常显示为 `GlobalAveragePool`），确认它的输入是 `[1, 576, 7, 7]`。
   这个输入张量就是 backbone 的输出。Netron 用于查看 ONNX 图，切分由本脚本完成。
3. 运行 `/usr/bin/python3 models/MobileNetV3-Small/02_model_convert/03-onnx_slice.py`，
   导出 backbone 和分类头两个 ONNX 文件。
4. 在 Netron 分别打开导出的文件，确认 backbone 输出和分类头输入形状一致。
   再在目标设备评估端到端延迟、内存和任务精度。

backbone 包含全部 `features` 层；分类头包含全局平均池化、展平和分类器。
图像缩放、裁剪和归一化仍由 ONNX 图外的预处理完成。
"""

from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small

project_dir = Path(__file__).resolve().parent
checkpoint_path = project_dir / "pt" / "mobilenet_v3_small-047dcff4.pt"
onnx_dir = project_dir / "onnx"
onnx_dir.mkdir(parents=True, exist_ok=True)

model = mobilenet_v3_small(weights=None)
model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
model.eval()

backbone = model.features
classification_head = nn.Sequential(
    model.avgpool,
    nn.Flatten(1),
    model.classifier,
)
backbone.eval()
classification_head.eval()

example_input = torch.zeros(1, 3, 224, 224, dtype=torch.float32)
with torch.no_grad():
    backbone_features = backbone(example_input)

print("Backbone 输出：{}".format(tuple(backbone_features.shape)))

torch.onnx.export(
    backbone,
    example_input,
    onnx_dir / "mobilenet_v3_small_backbone.onnx",
    input_names=["images"],
    output_names=["features"],
    opset_version=17,
    do_constant_folding=True,
    dynamo=False,
)

torch.onnx.export(
    classification_head,
    backbone_features,
    onnx_dir / "mobilenet_v3_small_classification_head.onnx",
    input_names=["features"],
    output_names=["logits"],
    opset_version=17,
    do_constant_folding=True,
    dynamo=False,
)

print("Backbone 和分类头 ONNX 已导出到：{}".format(onnx_dir))
