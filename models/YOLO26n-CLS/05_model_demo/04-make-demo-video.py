"""Create a short clip from the real C++ demo output and its input image."""

import re
from pathlib import Path

import cv2
import numpy as np

project_dir = Path(__file__).resolve().parent
metrics_dir = project_dir / "metrics"
sample_path = project_dir / "baseline" / "assets" / "sample.jpg"
video_path = project_dir / "demo.mp4"
results = []

for version, output_name in [
    ("Baseline", "baseline-output.txt"),
    ("V1 CUDA Graph", "v1-output.txt"),
]:
    output = (metrics_dir / output_name).read_text()
    sections = re.split(r"(?=\[(?:CUDA|OpenCV) preprocessing \+ TensorRT\])", output)
    for section in sections:
        if not section.strip():
            continue
        preprocess_type = section.splitlines()[0].split("preprocessing", 1)[0].strip("[] ")
        result = re.search(r"Result: (.+)", section).group(1)
        preprocess_ms = re.search(r"Preprocess: ([0-9.]+ ms)", section).group(1)
        inference_ms = re.search(r"Inference: ([0-9.]+ ms)", section).group(1)
        end_to_end_ms = re.search(r"End-to-end: ([0-9.]+ ms)", section).group(1)
        results.append((version, preprocess_type, result, preprocess_ms, inference_ms, end_to_end_ms))

if len(results) != 4:
    raise RuntimeError("需要 baseline 与 V1 的四条预处理结果，请先运行 03-system-validation.sh")

sample = cv2.imread(str(sample_path), cv2.IMREAD_COLOR)
if sample is None:
    raise FileNotFoundError(sample_path)
frame_width, frame_height = 1280, 720
fps = 15
writer = cv2.VideoWriter(
    str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height)
)
if not writer.isOpened():
    raise RuntimeError("OpenCV 无法创建 MP4 视频")

scale = min(540 / sample.shape[1], 480 / sample.shape[0])
sample = cv2.resize(sample, (int(sample.shape[1] * scale), int(sample.shape[0] * scale)))
for version, preprocess_type, result, preprocess_ms, inference_ms, end_to_end_ms in results:
    frame = np.full((frame_height, frame_width, 3), (24, 28, 35), dtype=np.uint8)
    cv2.putText(frame, "MicroZoo | YOLO26n-cls TensorRT", (55, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(frame, "Jetson Orin Nano Super  |  FP16 engine", (58, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (170, 190, 210), 1)
    x, y = 55, 165
    frame[y:y + sample.shape[0], x:x + sample.shape[1]] = sample
    cv2.rectangle(frame, (x - 2, y - 2), (x + sample.shape[1] + 1, y + sample.shape[0] + 1), (90, 105, 125), 2)
    cv2.putText(frame, version + "  |  " + preprocess_type, (650, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (80, 205, 255), 2)
    cv2.putText(frame, "Top-1: " + result, (650, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (120, 240, 150), 2)
    cv2.putText(frame, "Preprocess:  " + preprocess_ms, (650, 355), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (235, 235, 235), 2)
    cv2.putText(frame, "Inference:   " + inference_ms, (650, 415), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (235, 235, 235), 2)
    cv2.putText(frame, "End-to-end:  " + end_to_end_ms, (650, 475), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (235, 235, 235), 2)
    cv2.putText(frame, "Actual output from 03-system-validation.sh", (55, 675), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170, 190, 210), 1)
    for _ in range(fps * 3):
        writer.write(frame)

writer.release()
print("Demo video: {} ({:.2f} MiB)".format(video_path, video_path.stat().st_size / 1024 ** 2))
