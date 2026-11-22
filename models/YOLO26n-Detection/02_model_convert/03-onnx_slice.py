"""Export YOLO26n's backbone+neck and Detect module as separate ONNX graphs."""

from pathlib import Path

import onnx
import torch
from torch import nn
from ultralytics import YOLO, __version__


class FeatureExtractorExport(nn.Module):
    def __init__(self, layers, saved_layers, detect_inputs):
        super().__init__()
        self.layers = nn.ModuleList(layers[:-1])
        self.saved_layers = set(saved_layers)
        self.detect_inputs = detect_inputs

    def forward(self, images):
        outputs = []
        value = images
        for layer in self.layers:
            source = layer.f
            if source != -1:
                if isinstance(source, int):
                    value = outputs[source]
                else:
                    value = [value if index == -1 else outputs[index] for index in source]
            value = layer(value)
            outputs.append(value if layer.i in self.saved_layers else None)
        return tuple(outputs[index] for index in self.detect_inputs)


class DetectHeadExport(nn.Module):
    def __init__(self, detect_head):
        super().__init__()
        self.detect_head = detect_head

    def forward(self, *features):
        output = self.detect_head(list(features))
        return output[0] if isinstance(output, (tuple, list)) else output


project_dir = Path(__file__).resolve().parent
checkpoint_path = project_dir.parent / "01_model_train" / "checkpoints" / "yolo26n.pt"
onnx_dir = project_dir / "onnx"
onnx_dir.mkdir(parents=True, exist_ok=True)
feature_extractor_path = onnx_dir / "yolo26n_backbone_neck.onnx"
detect_head_path = onnx_dir / "yolo26n_detect_head.onnx"

if __version__ != "8.4.155":
    raise RuntimeError("需要 ultralytics==8.4.155，当前版本为 {}".format(__version__))
if not checkpoint_path.is_file():
    raise FileNotFoundError("先运行 01_model_train/01-download_weights.py")

network = YOLO(str(checkpoint_path)).model.eval()
layers = list(network.model)
detect_head = layers[-1]
detect_inputs = list(detect_head.f) if isinstance(detect_head.f, (list, tuple)) else [detect_head.f]
if len(layers) != 24 or detect_head.__class__.__name__ != "Detect":
    raise RuntimeError("预期 YOLO26n 的 23 个骨干/Neck 层和最后一个 Detect 层，当前结构已变化")
if detect_inputs != [16, 19, 22]:
    raise RuntimeError("预期 Detect 输入来自 model[16, 19, 22]，当前输入为 {}".format(detect_inputs))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
network = network.to(device)
detect_head.export = True
detect_head.format = "onnx"
detect_head.dynamic = False
feature_names = ["p3", "p4", "p5"]
feature_shapes = [(1, 64, 80, 80), (1, 128, 40, 40), (1, 256, 20, 20)]
example_input = torch.zeros((1, 3, 640, 640), device=device)
feature_extractor = FeatureExtractorExport(layers, network.save, detect_inputs).to(device).eval()
detect_head_export = DetectHeadExport(detect_head).to(device).eval()

with torch.inference_mode():
    features = feature_extractor(example_input)
    full_output = network(example_input)
    split_output = detect_head_export(*[feature.clone() for feature in features])
    if isinstance(full_output, (tuple, list)):
        full_output = full_output[0]
    actual_feature_shapes = [tuple(feature.shape) for feature in features]
    if actual_feature_shapes != feature_shapes:
        raise RuntimeError("P3/P4/P5 shape 变化：{}".format(actual_feature_shapes))
    if tuple(split_output.shape) != (1, 300, 6):
        raise RuntimeError("Detect head 输出 shape 变化：{}".format(tuple(split_output.shape)))
    max_error = (full_output - split_output).abs().max().item()
    if not torch.allclose(full_output, split_output, rtol=1e-5, atol=1e-5):
        raise RuntimeError("完整模型与切分模型输出不一致，最大绝对误差：{:.8g}".format(max_error))

print("切分点：YOLO26n model[23] Detect 之前")
print("特征提取段：model[0:23]（骨干网络 + Neck）")
for name, index, shape in zip(feature_names, detect_inputs, actual_feature_shapes):
    print("Detect 输入 {} <- model[{}]，shape={}".format(name, index, shape))
print("Detect head 输出：{}，完整模型与切分模型最大绝对误差：{:.8g}".format(
    tuple(split_output.shape), max_error
))

torch.onnx.export(
    feature_extractor,
    example_input,
    feature_extractor_path,
    input_names=["images"],
    output_names=feature_names,
    opset_version=17,
    do_constant_folding=True,
    dynamo=False,
)
torch.onnx.export(
    detect_head_export,
    tuple(feature.clone() for feature in features),
    detect_head_path,
    input_names=feature_names,
    output_names=["predictions"],
    opset_version=17,
    do_constant_folding=True,
    dynamo=False,
)

for onnx_path in [feature_extractor_path, detect_head_path]:
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("ONNX：{} ({:.2f} MiB)".format(onnx_path.name, onnx_path.stat().st_size / 1024 ** 2))
    for tensor in list(onnx_model.graph.input) + list(onnx_model.graph.output):
        dimensions = [dimension.dim_value for dimension in tensor.type.tensor_type.shape.dim]
        print("  {} {}".format(tensor.name, dimensions))
