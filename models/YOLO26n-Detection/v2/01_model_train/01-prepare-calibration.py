import random
import shutil
from pathlib import Path

import yaml

project_dir = Path(__file__).resolve().parent
model_dir = project_dir.parent
source_model_dir = model_dir.parent / "v1" / "01_model_train"
source_dataset = source_model_dir / "datasets" / "coco-val-32"
source_manifest = source_dataset / "selection.txt"
checkpoint_path = project_dir / "checkpoints" / "yolo26n.pt"
dataset_dir = project_dir / "datasets"
calibration_image_dir = dataset_dir / "calibration" / "images" / "val"
holdout_image_dir = dataset_dir / "holdout" / "images" / "val"
holdout_label_dir = dataset_dir / "holdout" / "labels" / "val"
holdout_yaml = dataset_dir / "holdout" / "data.yaml"
calibration_manifest = dataset_dir / "calibration_selection.txt"
holdout_manifest = dataset_dir / "holdout_selection.txt"
selection_seed = 20261122
calibration_count = 24
holdout_count = 8

# 使用 v1 生成并固定的 COCO val32 清单；随机种子和数量固定，确保校准/留出拆分可复现。
if not checkpoint_path.is_file():
    raise FileNotFoundError("请按 v2/README.md 将 v1 checkpoint 放到 {}".format(checkpoint_path))
if not source_manifest.is_file():
    raise FileNotFoundError("先运行 v1/01_model_train/00-prepare_coco_val.py")

selected_names = [
    line.split()[1]
    for line in source_manifest.read_text().splitlines()
    if line and not line.startswith("#")
]
if len(selected_names) != calibration_count + holdout_count:
    raise RuntimeError("预期 v1 的固定数据集有 32 张图片，实际为 {} 张".format(len(selected_names)))

random.Random(selection_seed).shuffle(selected_names)
# 校准集只用于估计激活范围；留出集保留标签，供后续任务精度验证使用。
calibration_names = sorted(selected_names[:calibration_count])
holdout_names = sorted(selected_names[calibration_count:])
source_image_dir = source_dataset / "images" / "val"
source_label_dir = source_dataset / "labels" / "val"

for image_name in calibration_names:
    source_image = source_image_dir / image_name
    if not source_image.is_file():
        raise FileNotFoundError("缺少校准图片：{}".format(source_image))
    calibration_image_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_image, calibration_image_dir / image_name)

for image_name in holdout_names:
    source_image = source_image_dir / image_name
    source_label = source_label_dir / (Path(image_name).stem + ".txt")
    if not source_image.is_file() or not source_label.is_file():
        raise FileNotFoundError("留出图片或标签缺失：{} / {}".format(source_image, source_label))
    holdout_image_dir.mkdir(parents=True, exist_ok=True)
    holdout_label_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_image, holdout_image_dir / image_name)
    shutil.copy2(source_label, holdout_label_dir / source_label.name)

source_data = yaml.safe_load((source_dataset / "data.yaml").read_text())
source_data["path"] = str((dataset_dir / "holdout").resolve())
source_data["train"] = "images/val"
source_data["val"] = "images/val"
holdout_yaml.parent.mkdir(parents=True, exist_ok=True)
holdout_yaml.write_text(yaml.safe_dump(source_data, allow_unicode=True, sort_keys=False))

dataset_dir.mkdir(parents=True, exist_ok=True)
calibration_manifest.write_text(
    "# seed={} count={} purpose=PTQ calibration\n{}\n".format(
        selection_seed, len(calibration_names), "\n".join(calibration_names)
    )
)
holdout_manifest.write_text(
    "# seed={} count={} purpose=held-out task validation\n{}\n".format(
        selection_seed, len(holdout_names), "\n".join(holdout_names)
    )
)

print("模型权重：{}".format(checkpoint_path))
print("PTQ 校准图片：{} 张，清单：{}".format(len(calibration_names), calibration_manifest))
print("留出验证图片：{} 张，清单：{}".format(len(holdout_names), holdout_manifest))
print("留出数据配置：{}".format(holdout_yaml))
print("拆分随机种子：{}".format(selection_seed))
