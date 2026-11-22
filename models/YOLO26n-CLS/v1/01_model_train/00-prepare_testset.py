import shutil
from pathlib import Path

project_dir = Path(__file__).resolve().parent
dataset_dir = project_dir / "datasets" / "imagenette2-160" / "val"
output_dir = project_dir / "datasets" / "imagenette2-small" / "val"
classes = [
    "n01440764",
    "n02102040",
    "n02979186",
    "n03000684",
    "n03028079",
    "n03394916",
    "n03417042",
    "n03425413",
    "n03445777",
    "n03888257",
]

if not dataset_dir.is_dir():
    raise FileNotFoundError("先运行 00-download_imagenette.sh 下载 Imagenette2-160")
if output_dir.parent.exists():
    shutil.rmtree(output_dir.parent)
output_dir.mkdir(parents=True)
manifest_lines = ["ImageNette2-160 val subset; first 10 lexically sorted images per class."]
for class_name in classes:
    source_images = sorted(path for path in (dataset_dir / class_name).iterdir() if path.is_file())
    if len(source_images) < 10:
        raise RuntimeError("类别 {} 少于 10 张图片".format(class_name))
    class_dir = output_dir / class_name
    class_dir.mkdir()
    for image_path in source_images[:10]:
        shutil.copy2(image_path, class_dir / image_path.name)
        manifest_lines.append("{}/{}".format(class_name, image_path.name))

sample_image = output_dir / "n01440764" / sorted(path.name for path in (output_dir / "n01440764").iterdir())[0]
sample_destinations = [
    project_dir / "sample.jpg",
    project_dir.parent / "03_model_validate" / "sample.jpg",
    project_dir.parent / "05_model_demo" / "assets" / "sample.jpg",
]
for destination in sample_destinations:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sample_image, destination)

manifest_path = output_dir.parent / "selection.txt"
manifest_path.write_text("\n".join(manifest_lines) + "\n")
print("ImageNette 固定验证子集：{} 张图片，{} 个类别".format(len(manifest_lines) - 1, len(classes)))
print("验证集目录：{}".format(output_dir.resolve()))
print("样本清单：{}".format(manifest_path.resolve()))
print("固定样例来源：{}".format(sample_image.resolve()))
for destination in sample_destinations:
    print("固定样例副本：{}".format(destination.resolve()))
