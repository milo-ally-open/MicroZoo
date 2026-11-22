from pathlib import Path

import cv2
import numpy as np
from ultralytics import __version__
from ultralytics.data.augment import LetterBox

project_dir = Path(__file__).resolve().parent
model_dir = project_dir.parent
calibration_image_dir = model_dir / "01_model_train" / "datasets" / "calibration" / "images" / "val"
output_dir = project_dir / "calibration"
output_path = output_dir / "yolo26n_calibration.npz"
image_size = 640
expected_image_count = 24

# stage 01 已经按固定清单拆分 calibration/holdout；这里只处理 calibration 图片，
# 并生成模型可直接读取的输入 tensor，不在此处重新抽样或下载数据。
if __version__ != "8.4.155":
    raise RuntimeError("需要 ultralytics==8.4.155，当前版本为 {}".format(__version__))
if not calibration_image_dir.is_dir():
    raise FileNotFoundError("先运行 ../01_model_train/01-prepare-calibration.py")

image_paths = sorted(calibration_image_dir.glob("*.jpg"))
if len(image_paths) != expected_image_count:
    raise RuntimeError("预期 {} 张校准图片，实际为 {} 张".format(expected_image_count, len(image_paths)))

letterbox = LetterBox(new_shape=(image_size, image_size), auto=False, scaleup=True)
calibration_batches = []
for image_path in image_paths:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("无法读取校准图片：{}".format(image_path))
    image = letterbox(image=image)
    # OpenCV 读入的是 BGR；模型使用 RGB、NCHW、FP32，像素归一化到 [0,1]。
    image = image[:, :, ::-1].transpose(2, 0, 1)
    calibration_batches.append(np.ascontiguousarray(image[None], dtype=np.float32) / 255.0)

calibration_data = np.concatenate(calibration_batches, axis=0)
if calibration_data.shape != (expected_image_count, 3, image_size, image_size):
    raise RuntimeError("校准输入 shape 异常：{}".format(calibration_data.shape))

# 压缩后的 NPZ 既保存校准输入，也让后续脚本无需重复图像预处理。
output_dir.mkdir(parents=True, exist_ok=True)
np.savez_compressed(output_path, images=calibration_data)
print("校准输入：{}".format(output_path))
print("ONNX 输入名：images")
print("数据 shape：{}，dtype={}，范围=[{:.6f}, {:.6f}]".format(calibration_data.shape, calibration_data.dtype, calibration_data.min(), calibration_data.max()))
print("图片数：{}；预处理：LetterBox(640, auto=False)、RGB、NCHW、FP32 [0,1]".format(len(image_paths)))
