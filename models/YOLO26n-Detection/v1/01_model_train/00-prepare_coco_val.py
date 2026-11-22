import json
import random
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import yaml

project_dir = Path(__file__).resolve().parent
dataset_dir = project_dir / "datasets" / "coco-val-32"
image_dir = dataset_dir / "images" / "val"
label_dir = dataset_dir / "labels" / "val"
manifest_path = dataset_dir / "selection.txt"
data_yaml_path = dataset_dir / "data.yaml"
sample_count = 32
selection_seed = 20261121
annotation_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
image_url = "http://images.cocodataset.org/val2017/{}"

if manifest_path.is_file() and data_yaml_path.is_file():
    selected_names = [line.split()[1] for line in manifest_path.read_text().splitlines()[1:]]
    if len(selected_names) == sample_count and all((image_dir / name).is_file() for name in selected_names):
        print("COCO val 子集已准备：{}，图片数：{}".format(dataset_dir, sample_count))
        print("清单：{}".format(manifest_path))
        raise SystemExit(0)

image_dir.mkdir(parents=True, exist_ok=True)
label_dir.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(prefix="microzoo-coco-") as temporary_dir:
    archive_path = Path(temporary_dir) / "annotations_trainval2017.zip"
    annotation_path = Path(temporary_dir) / "instances_val2017.json"
    print("下载 COCO 2017 官方标注包（只提取 instances_val2017.json）...", flush=True)
    with urllib.request.urlopen(annotation_url) as response, archive_path.open("wb") as output_file:
        total_bytes = int(response.headers.get("Content-Length", "0"))
        downloaded_bytes = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output_file.write(chunk)
            downloaded_bytes += len(chunk)
            if downloaded_bytes % (32 * 1024 * 1024) < len(chunk) or downloaded_bytes == total_bytes:
                print("标注包下载：{:.0f}/{:.0f} MiB".format(downloaded_bytes / 1024 ** 2, total_bytes / 1024 ** 2), flush=True)
    with zipfile.ZipFile(archive_path) as archive:
        with archive.open("annotations/instances_val2017.json") as source:
            annotation_path.write_bytes(source.read())

    annotations = json.loads(annotation_path.read_text())

categories = sorted(annotations["categories"], key=lambda item: item["id"])
category_to_index = {category["id"]: index for index, category in enumerate(categories)}
images_by_id = {image["id"]: image for image in annotations["images"]}
boxes_by_image = {}
for annotation in annotations["annotations"]:
    if annotation.get("iscrowd", 0) or annotation["category_id"] not in category_to_index:
        continue
    x, y, width, height = annotation["bbox"]
    if width <= 0 or height <= 0:
        continue
    boxes_by_image.setdefault(annotation["image_id"], []).append(annotation)

candidates = sorted(
    (image_id for image_id in boxes_by_image if image_id in images_by_id),
    key=lambda image_id: images_by_id[image_id]["file_name"],
)
random.Random(selection_seed).shuffle(candidates)
selected_ids = candidates[:sample_count]
if len(selected_ids) != sample_count:
    raise RuntimeError("COCO val 标注中可用图片不足 {} 张".format(sample_count))

manifest_lines = [
    "# COCO 2017 val2017 fixed subset; seed={}; images={}".format(selection_seed, sample_count),
    "# image_id file_name non_crowd_boxes",
]
for image_id in selected_ids:
    image = images_by_id[image_id]
    file_name = image["file_name"]
    image_path = image_dir / file_name
    if not image_path.is_file():
        print("下载图片：{}".format(file_name), flush=True)
        urllib.request.urlretrieve(image_url.format(file_name), image_path)

    label_path = label_dir / (Path(file_name).stem + ".txt")
    label_lines = []
    for annotation in boxes_by_image[image_id]:
        x, y, width, height = annotation["bbox"]
        center_x = (x + width / 2.0) / image["width"]
        center_y = (y + height / 2.0) / image["height"]
        normalized_width = width / image["width"]
        normalized_height = height / image["height"]
        label_lines.append("{} {:.8f} {:.8f} {:.8f} {:.8f}".format(
            category_to_index[annotation["category_id"]],
            center_x,
            center_y,
            normalized_width,
            normalized_height,
        ))
    label_path.write_text("\n".join(label_lines) + "\n")
    manifest_lines.append("{} {} {}".format(image_id, file_name, len(label_lines)))

manifest_path.write_text("\n".join(manifest_lines) + "\n")
data_yaml_path.write_text(yaml.safe_dump({
    "path": str(dataset_dir),
    "train": "images/val",
    "val": "images/val",
    "names": [category["name"] for category in categories],
}, allow_unicode=True, sort_keys=False))

print("COCO 官方 val2017 固定子集：{} 张图片".format(sample_count))
print("选择随机种子：{}".format(selection_seed))
print("类别：{}".format(len(categories)))
print("标注：{} 个非 crowd 框".format(sum(len(boxes_by_image[image_id]) for image_id in selected_ids)))
print("数据集配置：{}".format(data_yaml_path))
print("样本清单：{}".format(manifest_path))
