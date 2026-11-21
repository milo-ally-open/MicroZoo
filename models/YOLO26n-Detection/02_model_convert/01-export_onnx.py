import shutil
from pathlib import Path

import onnx
from ultralytics import YOLO, __version__

project_dir = Path(__file__).resolve().parent
source_checkpoint = project_dir.parent / "01_model_train" / "checkpoints" / "yolo26n.pt"
checkpoint_path = project_dir / "pt" / "yolo26n.pt"
onnx_dir = project_dir / "onnx"
onnx_path = onnx_dir / "yolo26n.onnx"

if __version__ != "8.4.155":
    raise RuntimeError("需要 ultralytics==8.4.155，当前版本为 {}".format(__version__))
if not source_checkpoint.is_file():
    raise FileNotFoundError("先运行 01_model_train/01-download_weights.py")

checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
onnx_dir.mkdir(parents=True, exist_ok=True)
shutil.copy2(source_checkpoint, checkpoint_path)
model = YOLO(str(checkpoint_path))
exported_path = Path(model.export(
    format="onnx",
    imgsz=640,
    batch=1,
    dynamic=False,
    opset=17,
    simplify=False,
    nms=False,
    device=0,
))
if exported_path.resolve() != onnx_path.resolve():
    shutil.copy2(exported_path, onnx_path)
    exported_path.unlink()

onnx_model = onnx.load(str(onnx_path))
onnx.checker.check_model(onnx_model)
print("权重副本：{}".format(checkpoint_path))
print("ONNX：{} ({:.2f} MiB)".format(onnx_path, onnx_path.stat().st_size / 1024 ** 2))
print("导出配置：固定 batch=1、640×640、opset=17、原始检测头（不含 NMS）")
for tensor in onnx_model.graph.input:
    dims = [dimension.dim_value for dimension in tensor.type.tensor_type.shape.dim]
    print("输入：{}，shape={}，elem_type={}".format(tensor.name, dims, tensor.type.tensor_type.elem_type))
for tensor in onnx_model.graph.output:
    dims = [dimension.dim_value for dimension in tensor.type.tensor_type.shape.dim]
    print("输出：{}，shape={}，elem_type={}".format(tensor.name, dims, tensor.type.tensor_type.elem_type))
