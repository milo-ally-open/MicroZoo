# Load Conda Pillow before system OpenCV to avoid their JPEG library load-order conflict on Jetson.
import json
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor
import cv2


MODEL_ID = "google/owlv2-base-patch16-ensemble"
MODEL_REVISION = "cfd3195ba4ea9592eec887ded089f4c08eff231d"
MODEL_DIR = Path(__file__).resolve().parent / "checkpoints" / "owlv2-base-patch16-ensemble"
IMAGE_PATH = Path(__file__).resolve().parent / "assets" / "coco_000000039769.jpg"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
PROMPTS = [
    "a photo of a cat",
    "a photo of a dog",
    "a photo of a person",
    "a photo of a remote control",
]
SCORE_THRESHOLD = 0.1

if not torch.cuda.is_available():
    raise RuntimeError("OWLv2 分析基线要求 Jetson CUDA PyTorch。")
if not (MODEL_DIR / "model.safetensors").is_file():
    raise FileNotFoundError("请先运行 02-download_model.py 下载固定 revision 的权重。")
if not IMAGE_PATH.is_file():
    raise FileNotFoundError("测试图像不存在：{}".format(IMAGE_PATH))

device = torch.device("cuda:0")
image = Image.open(IMAGE_PATH).convert("RGB")
processor = Owlv2Processor.from_pretrained(MODEL_DIR, local_files_only=True)
model = Owlv2ForObjectDetection.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
    use_safetensors=True,
    torch_dtype=torch.float16,
).eval().to(device)

start = time.perf_counter()
pipeline_start = start
inputs = processor(text=[PROMPTS], images=image, return_tensors="pt")
preprocess_ms = (time.perf_counter() - start) * 1000.0
input_shapes = {name: list(value.shape) for name, value in inputs.items()}
start = time.perf_counter()
inputs = {name: value.to(device) for name, value in inputs.items()}
inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.float16)
torch.cuda.synchronize()
transfer_ms = (time.perf_counter() - start) * 1000.0

torch.cuda.reset_peak_memory_stats(device)
start = time.perf_counter()
with torch.inference_mode():
    outputs = model(**inputs)
torch.cuda.synchronize()
inference_ms = (time.perf_counter() - start) * 1000.0
output_shapes = {
    name: list(getattr(outputs, name).shape)
    for name in ["logits", "pred_boxes"]
    if getattr(outputs, name, None) is not None
}

target_sizes = torch.tensor([[image.height, image.width]])
start = time.perf_counter()
detections = processor.post_process_object_detection(
    outputs=outputs,
    target_sizes=target_sizes,
    threshold=SCORE_THRESHOLD,
)[0]

boxes = detections["boxes"].cpu().tolist()
scores = detections["scores"].cpu().tolist()
labels = detections["labels"].cpu().tolist()
records = []
for box, score, label in zip(boxes, scores, labels):
    records.append({
        "prompt": PROMPTS[label],
        "score": float(score),
        "box_xyxy": [float(value) for value in box],
    })
records.sort(key=lambda row: row["score"], reverse=True)
postprocess_ms = (time.perf_counter() - start) * 1000.0
end_to_end_ms = (time.perf_counter() - pipeline_start) * 1000.0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
result_path = OUTPUT_DIR / "first-inference.json"
result_path.write_text(json.dumps({
    "model_id": MODEL_ID,
    "model_revision": MODEL_REVISION,
    "image": str(IMAGE_PATH.relative_to(Path(__file__).resolve().parent)),
    "image_size_wh": [image.width, image.height],
    "prompts": PROMPTS,
    "score_threshold": SCORE_THRESHOLD,
    "device": torch.cuda.get_device_name(0),
    "model_dtype": "float16",
    "input_shapes": input_shapes,
    "output_shapes": output_shapes,
    "preprocess_ms": preprocess_ms,
    "input_transfer_ms": transfer_ms,
    "inference_ms": inference_ms,
    "postprocess_ms": postprocess_ms,
    "pipeline_end_to_end_ms": end_to_end_ms,
    "pytorch_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024 * 1024),
    "detections": records,
}, indent=2, ensure_ascii=False), encoding="utf-8")

# Save a visual preview for this one-image smoke check; it is not a task metric.
preview = cv2.cvtColor(cv2.imread(str(IMAGE_PATH)), cv2.COLOR_BGR2RGB)
for row in records[:30]:
    x1, y1, x2, y2 = [int(round(value)) for value in row["box_xyxy"]]
    text = "{} {:.2f}".format(row["prompt"], row["score"])
    cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(preview, text, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
preview_path = OUTPUT_DIR / "first-inference.jpg"
cv2.imwrite(str(preview_path), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))

print("Model:", MODEL_ID)
print("Revision:", MODEL_REVISION)
print("Image:", IMAGE_PATH.name, "({}x{})".format(image.width, image.height))
print("Input shapes:", input_shapes)
print("Output shapes:", output_shapes)
print("Preprocess: {:.2f} ms".format(preprocess_ms))
print("Input H2D: {:.2f} ms".format(transfer_ms))
print("PyTorch FP16 inference: {:.2f} ms".format(inference_ms))
print("Postprocess: {:.2f} ms".format(postprocess_ms))
print("Pipeline end-to-end: {:.2f} ms".format(end_to_end_ms))
print("PyTorch peak allocated: {:.0f} MiB".format(torch.cuda.max_memory_allocated(device) / (1024 * 1024)))
print("Detections above {:.2f}: {}".format(SCORE_THRESHOLD, len(records)))
for row in records[:10]:
    print("{} {:.3f} {}".format(row["prompt"], row["score"], row["box_xyxy"]))
print("JSON:", result_path)
print("Preview:", preview_path)
