from pathlib import Path

from ultralytics import YOLO, __version__
from ultralytics.utils.downloads import attempt_download_asset

project_dir = Path(__file__).resolve().parent
checkpoint_path = project_dir / "checkpoints" / "yolo26n-cls.pt"

if __version__ != "8.4.155":
    raise RuntimeError("需要 ultralytics==8.4.155，当前版本为 {}".format(__version__))

checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
attempt_download_asset(str(checkpoint_path), release="v8.4.0", progress=True)
model = YOLO(str(checkpoint_path))
if model.task != "classify" or len(model.names) != 1000:
    raise RuntimeError("下载的 checkpoint 不是 ImageNet-1K YOLO26 分类模型")

print("权重：{}".format(checkpoint_path))
print("来源：Ultralytics assets v8.4.0 / yolo26n-cls.pt")
print("任务：{}，类别数：{}".format(model.task, len(model.names)))
print("文件大小：{:.2f} MiB".format(checkpoint_path.stat().st_size / 1024 ** 2))
