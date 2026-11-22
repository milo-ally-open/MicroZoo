# Export OWLv2 into composable vision, text, and text-image matching ONNX graphs.
# Run with: conda run -n microzoo-owlv2-v1 python 02-export_split_onnx.py
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
OPSET = 17
PROMPTS = [
    "a photo of a cat",
    "a photo of a dog",
    "a photo of a person",
    "a photo of a remote control",
]


class TextBranch(nn.Module):
    # Text prompts are encoded once and may be cached while the prompt list is unchanged.
    def __init__(self, detector):
        super().__init__()
        self.detector = detector

    def forward(self, input_ids, attention_mask):
        text_embeds = self.detector.owlv2.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        # OWLv2's main forward normalizes projected CLIP text features here.
        text_embeds = text_embeds / torch.linalg.norm(text_embeds, ord=2, dim=-1, keepdim=True)
        query_mask = input_ids[:, 0] > 0
        return text_embeds, query_mask


class VisionBranch(nn.Module):
    # The vision graph includes the visual backbone and all image-only detection heads.
    def __init__(self, detector):
        super().__init__()
        self.detector = detector

    def forward(self, pixel_values):
        feature_map = self.detector.image_embedder(pixel_values)[0]
        batch_size = feature_map.shape[0]
        num_patches = feature_map.shape[1] * feature_map.shape[2]
        hidden_size = feature_map.shape[-1]
        image_feats = feature_map.reshape(batch_size, num_patches, hidden_size)

        image_class_embeds = self.detector.class_head.dense0(image_feats)
        image_class_embeds = image_class_embeds / (
            torch.linalg.norm(image_class_embeds, ord=2, dim=-1, keepdim=True) + 1e-6
        )
        logit_shift = self.detector.class_head.logit_shift(image_feats)
        logit_scale = self.detector.class_head.elu(self.detector.class_head.logit_scale(image_feats)) + 1
        pred_boxes = self.detector.box_predictor(image_feats, feature_map)
        objectness_logits = self.detector.objectness_predictor(image_feats)
        return image_class_embeds, logit_shift, logit_scale, pred_boxes, objectness_logits


class MatchBranch(nn.Module):
    # This small graph combines cached text features with each image's visual features.
    def forward(self, image_class_embeds, logit_shift, logit_scale, text_embeds, query_mask):
        text_embeds = text_embeds / (
            torch.linalg.norm(text_embeds, ord=2, dim=-1, keepdim=True) + 1e-6
        )
        logits = torch.matmul(image_class_embeds, text_embeds.transpose(0, 1))
        logits = (logits + logit_shift) * logit_scale
        logits = torch.where(
            query_mask.reshape(1, 1, -1),
            logits,
            torch.full_like(logits, torch.finfo(logits.dtype).min),
        )
        return logits.to(torch.float32)


if not torch.cuda.is_available():
    raise RuntimeError("ONNX 导出要求当前 Jetson 环境能使用 CUDA PyTorch。")
if not (MODEL_DIR / "model.safetensors").is_file():
    raise FileNotFoundError("模型权重不存在，请先完成 01_model_analysis 的模型下载。")
if not IMAGE_PATH.is_file():
    raise FileNotFoundError("切分验证用示例图片不存在：{}".format(IMAGE_PATH))

device = torch.device("cuda:0")
processor = Owlv2Processor.from_pretrained(MODEL_DIR, local_files_only=True)
detector = Owlv2ForObjectDetection.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
    use_safetensors=True,
    torch_dtype=torch.float16,
).eval().to(device)
text_branch = TextBranch(detector).eval()
vision_branch = VisionBranch(detector).eval()
match_branch = MatchBranch().eval()

image = Image.open(IMAGE_PATH).convert("RGB")
inputs = processor(text=[PROMPTS], images=image, return_tensors="pt")
input_ids = inputs["input_ids"].to(device)
pixel_values = inputs["pixel_values"].to(device=device, dtype=torch.float16)
attention_mask = inputs["attention_mask"].to(device)

# Verify that independently exported branches reconstruct the official forward pass.
print("正在核对完整模型与三段计算图的 FP16 输出……", flush=True)
with torch.inference_mode():
    reference = detector(
        input_ids=input_ids,
        pixel_values=pixel_values,
        attention_mask=attention_mask,
        return_dict=True,
    )
    query_embeds, query_mask = text_branch(input_ids, attention_mask)
    image_class_embeds, logit_shift, logit_scale, split_boxes, split_objectness = vision_branch(pixel_values)
    split_logits = match_branch(
        image_class_embeds, logit_shift, logit_scale, query_embeds, query_mask
    )

torch.cuda.synchronize()
logits_error = (split_logits - reference.logits).abs().max().item()
boxes_error = (split_boxes - reference.pred_boxes).abs().max().item()
objectness_error = (split_objectness - reference.objectness_logits).abs().max().item()
torch.testing.assert_close(split_logits, reference.logits, rtol=1e-3, atol=1e-3)
torch.testing.assert_close(split_boxes, reference.pred_boxes, rtol=1e-3, atol=1e-3)
torch.testing.assert_close(split_objectness, reference.objectness_logits, rtol=1e-3, atol=1e-3)
print("最大绝对误差：logits={:.8g}, pred_boxes={:.8g}, objectness={:.8g}".format(
    logits_error, boxes_error, objectness_error
))

ONNX_DIR.mkdir(parents=True, exist_ok=True)
print("开始导出三张独立 ONNX 图，opset {}。".format(OPSET), flush=True)

torch.onnx.export(
    text_branch,
    (input_ids, attention_mask),
    str(ONNX_DIR / "owlv2_text_fp16.onnx"),
    export_params=True,
    opset_version=OPSET,
    do_constant_folding=True,
    input_names=["input_ids", "attention_mask"],
    output_names=["text_embeds", "query_mask"],
    dynamic_axes={
        "input_ids": {0: "num_queries"},
        "attention_mask": {0: "num_queries"},
        "text_embeds": {0: "num_queries"},
        "query_mask": {0: "num_queries"},
    },
    dynamo=False,
)
print("已导出文本编码图。", flush=True)

torch.onnx.export(
    vision_branch,
    (pixel_values,),
    str(ONNX_DIR / "owlv2_vision_fp16.onnx"),
    export_params=True,
    opset_version=OPSET,
    do_constant_folding=True,
    input_names=["pixel_values"],
    output_names=[
        "image_class_embeds", "logit_shift", "logit_scale", "pred_boxes", "objectness_logits"
    ],
    dynamo=False,
)
print("已导出视觉编码与图像检测头。", flush=True)

torch.onnx.export(
    match_branch,
    (image_class_embeds, logit_shift, logit_scale, query_embeds, query_mask),
    str(ONNX_DIR / "owlv2_match_fp16.onnx"),
    export_params=True,
    opset_version=OPSET,
    do_constant_folding=True,
    input_names=[
        "image_class_embeds", "logit_shift", "logit_scale", "text_embeds", "query_mask"
    ],
    output_names=["logits"],
    dynamic_axes={
        "text_embeds": {0: "num_queries"},
        "query_mask": {0: "num_queries"},
        "logits": {2: "num_queries"},
    },
    dynamo=False,
)
print("已导出图文匹配图。", flush=True)

for onnx_path in sorted(ONNX_DIR.glob("owlv2_*_fp16.onnx")):
    print("{} ({:.1f} MiB)".format(onnx_path.name, onnx_path.stat().st_size / (1024 * 1024)))
