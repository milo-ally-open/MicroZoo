"""导出 YOLO26n-cls 的 backbone 和 classification head 两段 ONNX。

运行 02-check_layers.py 可确认 model[9] 输出 [1, 256, 7, 7]；model[10]
是 Classify 模块。Netron (https://netron.app) 可分别打开 full、backbone、head
ONNX 检查边界：backbone 的 features 输出要与 head 的 features 输入匹配。
"""

from pathlib import Path

import onnx
import torch
from PIL import Image
from torch import nn
from ultralytics import YOLO
from ultralytics.data.augment import classify_transforms


class ClassificationHeadExport(nn.Module):
    """让不同 Ultralytics 版本的分类头统一只导出概率张量。"""

    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, features):
        output = self.head(features)
        return output[0] if isinstance(output, (tuple, list)) else output


project_dir = Path(__file__).resolve().parent
checkpoint_path = project_dir / "pt" / "yolo26n-cls.pt"
onnx_dir = project_dir / "onnx"
onnx_dir.mkdir(parents=True, exist_ok=True)
model = YOLO(str(checkpoint_path)).model.eval()
layers = list(model.model)
if len(layers) != 11 or layers[-1].__class__.__name__ != "Classify":
    raise RuntimeError("预期 10 个 backbone 层和最后一个 Classify 层，实际网络结构已变化")

backbone = nn.Sequential(*layers[:10]).eval()
classification_head = layers[10].eval()
classification_head.export = True
classification_head_export = ClassificationHeadExport(classification_head).eval()
image_path = project_dir.parent / "01_model_train" / "sample.jpg"
example_input = classify_transforms(224)(Image.open(image_path).convert("RGB")).unsqueeze(0)
with torch.no_grad():
    backbone_features = backbone(example_input)
    full_output = model(example_input)
    if isinstance(full_output, (tuple, list)):
        full_output = full_output[0]
    split_output = classification_head_export(backbone_features.clone())
if tuple(backbone_features.shape) != (1, 256, 7, 7):
    raise RuntimeError("backbone 输出 shape 变化：{}".format(tuple(backbone_features.shape)))
if tuple(split_output.shape) != (1, 1000):
    raise RuntimeError("classification head 输出 shape 变化：{}".format(tuple(split_output.shape)))
print("切分点：model[10] Classify 之前")
print("Backbone 输出 / head 输入：{}".format(tuple(backbone_features.shape)))
print("Head 输出：{}，PyTorch full vs split 最大误差：{:.8g}".format(
    tuple(split_output.shape), (full_output - split_output).abs().max().item()
))

torch.onnx.export(
    backbone,
    example_input,
    onnx_dir / "yolo26n-cls_backbone.onnx",
    input_names=["images"],
    output_names=["features"],
    opset_version=17,
    do_constant_folding=True,
    dynamo=False,
)
torch.onnx.export(
    classification_head_export,
    backbone_features.clone(),
    onnx_dir / "yolo26n-cls_head.onnx",
    input_names=["features"],
    output_names=["probabilities"],
    opset_version=17,
    do_constant_folding=True,
    dynamo=False,
)

for onnx_path in [
    onnx_dir / "yolo26n-cls_backbone.onnx",
    onnx_dir / "yolo26n-cls_head.onnx",
]:
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("ONNX：{} ({:.2f} MiB)".format(onnx_path.name, onnx_path.stat().st_size / 1024 ** 2))
    for tensor in list(onnx_model.graph.input) + list(onnx_model.graph.output):
        dims = [dimension.dim_value for dimension in tensor.type.tensor_type.shape.dim]
        print("  {} {}".format(tensor.name, dims))
