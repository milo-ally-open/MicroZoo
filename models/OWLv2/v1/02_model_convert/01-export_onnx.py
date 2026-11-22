# Export the complete OWLv2 detector as a reference ONNX graph.
# Run with: conda run -n microzoo-owlv2-v1 python 01-export_onnx.py
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from transformers import Owlv2ForObjectDetection, Owlv2Processor


MODEL_ID = "google/owlv2-base-patch16-ensemble"
MODEL_REVISION = "cfd3195ba4ea9592eec887ded089f4c08eff231d"
STAGE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = STAGE_DIR / "01_model_analysis" / "checkpoints" / "owlv2-base-patch16-ensemble"
IMAGE_PATH = STAGE_DIR / "01_model_analysis" / "assets" / "coco_000000039769.jpg"
ONNX_DIR = Path(__file__).resolve().parent / "onnx"
ONNX_PATH = ONNX_DIR / "owlv2_full_fp16.onnx"
PROMPTS = [
    "a photo of a cat",
    "a photo of a dog",
    "a photo of a person",
    "a photo of a remote control",
]
OPSET = 17


class FullDetector(nn.Module):
    # Keep the official model's complete text-conditioned detection forward pass.
    def __init__(self, detector):
        super().__init__()
        self.detector = detector

    def forward(self, input_ids, pixel_values, attention_mask):
        outputs = self.detector(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return outputs.logits, outputs.pred_boxes, outputs.objectness_logits


if not torch.cuda.is_available():
    raise RuntimeError("ONNX 导出要求当前 Jetson 环境能使用 CUDA PyTorch。")
if not (MODEL_DIR / "model.safetensors").is_file():
    raise FileNotFoundError("模型权重不存在，请先完成 01_model_analysis 的模型下载。")
if not IMAGE_PATH.is_file():
    raise FileNotFoundError("导出用示例图片不存在：{}".format(IMAGE_PATH))

device = torch.device("cuda:0")
processor = Owlv2Processor.from_pretrained(MODEL_DIR, local_files_only=True)
detector = Owlv2ForObjectDetection.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
    use_safetensors=True,
    torch_dtype=torch.float16,
).eval().to(device)
wrapped_model = FullDetector(detector).eval()

image = Image.open(IMAGE_PATH).convert("RGB")
inputs = processor(text=[PROMPTS], images=image, return_tensors="pt")
input_ids = inputs["input_ids"].to(device)
pixel_values = inputs["pixel_values"].to(device=device, dtype=torch.float16)
attention_mask = inputs["attention_mask"].to(device)

ONNX_DIR.mkdir(parents=True, exist_ok=True)
print("模型：{} @ {}".format(MODEL_ID, MODEL_REVISION))
print("权重：{}".format(MODEL_DIR / "model.safetensors"))
print("导出精度：FP16；opset：{}".format(OPSET))
print("输入形状：input_ids={} pixel_values={} attention_mask={}".format(tuple(input_ids.shape), tuple(pixel_values.shape), tuple(attention_mask.shape)))
print("开始导出完整参考图；OWLv2 含大型视觉与文本 Transformer，导出可能需要几分钟。", flush=True)

with torch.inference_mode():
    torch.onnx.export(
        wrapped_model,
        (input_ids, pixel_values, attention_mask),
        str(ONNX_PATH),
        export_params=True,
        opset_version=OPSET,
        do_constant_folding=True,
        input_names=["input_ids", "pixel_values", "attention_mask"],
        output_names=["logits", "pred_boxes", "objectness_logits"],
        dynamic_axes={
            "input_ids": {0: "num_queries"},
            "attention_mask": {0: "num_queries"},
            "logits": {2: "num_queries"},
        },
        dynamo=False,
        verbose=False,
    )

torch.cuda.synchronize()
print("完整参考 ONNX：{} ({} MiB)".format(ONNX_PATH, ONNX_PATH.stat().st_size / (1024 * 1024)))
print("ONNX 输入：input_ids[Q,16]、pixel_values[1,3,960,960]、attention_mask[Q,16]")
print("ONNX 输出：logits[1,3600,Q]、pred_boxes[1,3600,4]、objectness_logits[1,3600]")
