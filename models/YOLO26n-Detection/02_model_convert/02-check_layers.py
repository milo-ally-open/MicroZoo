from pathlib import Path

import torch
from ultralytics import YOLO

project_dir = Path(__file__).resolve().parent
checkpoint_path = project_dir.parent / "01_model_train" / "checkpoints" / "yolo26n.pt"
if not checkpoint_path.is_file():
    raise FileNotFoundError("先运行 01_model_train/01-download_weights.py")

wrapper = YOLO(str(checkpoint_path))
network = wrapper.model.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
network.to(device)
captured_shapes = {}
hooks = []

for index, layer in enumerate(network.model):
    def capture_output(module, inputs, output, layer_index=index):
        if torch.is_tensor(output):
            captured_shapes[layer_index] = tuple(output.shape)
        elif isinstance(output, (tuple, list)):
            captured_shapes[layer_index] = [tuple(value.shape) for value in output if torch.is_tensor(value)]
        else:
            captured_shapes[layer_index] = type(output).__name__

    hooks.append(layer.register_forward_hook(capture_output))

with torch.inference_mode():
    output = network(torch.zeros(1, 3, 640, 640, device=device))
if torch.cuda.is_available():
    torch.cuda.synchronize()
for hook in hooks:
    hook.remove()

print("Checkpoint：{}".format(checkpoint_path))
print("设备：{}".format(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"))
print("输入： [1, 3, 640, 640]")
for index, layer in enumerate(network.model):
    source = getattr(layer, "f", -1)
    parameters = sum(parameter.numel() for parameter in layer.parameters())
    print("{:02d} from={:<14} {:<24} params={:<10,} output={}".format(
        index,
        str(source),
        layer.__class__.__name__,
        parameters,
        captured_shapes.get(index, "unknown"),
    ))
if isinstance(output, (tuple, list)):
    print("模型输出：{}".format([tuple(value.shape) for value in output if torch.is_tensor(value)]))
else:
    print("模型输出：{}".format(tuple(output.shape)))
