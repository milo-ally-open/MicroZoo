from pathlib import Path

import torch
from torchvision.models import mobilenet_v3_small

project_dir = Path(__file__).resolve().parent
checkpoint_path = project_dir / "pt" / "mobilenet_v3_small-047dcff4.pt"

model = mobilenet_v3_small(weights=None)
model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
model.eval()
x = torch.zeros(1, 3, 224, 224)

with torch.inference_mode():
    for index, layer in enumerate(model.features):
        x = layer(x)
        size_kib = x.numel() * x.element_size() / 1024
        module_types = " -> ".join(
            module.__class__.__name__
            for module in layer.modules()
            if module is not layer and not list(module.children())
        )
        print(
            "features[{}] ({}) [{}] -> {} ({:.1f} KiB)".format(
                index,
                layer.__class__.__name__,
                module_types,
                tuple(x.shape),
                size_kib,
            )
        )

    x = model.avgpool(x)
    print("avgpool ({}) -> {}".format(model.avgpool.__class__.__name__, tuple(x.shape)))

    x = torch.flatten(x, 1)
    print("flatten -> {}".format(tuple(x.shape)))

    for index, layer in enumerate(model.classifier):
        x = layer(x)
        print(
            "classifier[{}] ({}) -> {}".format(
                index,
                layer.__class__.__name__,
                tuple(x.shape),
            )
        )
