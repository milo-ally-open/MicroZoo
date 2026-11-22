import shutil
from pathlib import Path

import onnx
from ultralytics import YOLO, __version__

project_dir = Path(__file__).resolve().parent
checkpoint_path = project_dir / "pt" / "yolo26n.pt"
onnx_dir = project_dir / "onnx"
onnx_path = onnx_dir / "yolo26n_fp32.onnx"

if __version__ != "8.4.155":
    raise RuntimeError("需要 ultralytics==8.4.155，当前版本为 {}".format(__version__))
if not checkpoint_path.is_file():
    raise FileNotFoundError("先把 v1 的 yolo26n.pt 放到 {}".format(checkpoint_path))

# 先从 PyTorch checkpoint 导出完整 FP32 ONNX；后面的剪枝脚本只处理 ONNX，
# 这样可以把原始权重、剪枝图和部署 engine 各自保留为清楚的阶段产物。
onnx_dir.mkdir(parents=True, exist_ok=True)
model = YOLO(str(checkpoint_path))
exported_path = Path(model.export(
    format="onnx",
    imgsz=640,
    batch=1,
    dynamic=False,
    half=False,
    opset=17,
    simplify=False,
    nms=False,
    device=0,
))

# Ultralytics 默认把导出文件放在 checkpoint 附近；统一移动到 onnx/ 并固定文件名。
if exported_path.resolve() != onnx_path.resolve():
    shutil.copy2(exported_path, onnx_path)
    exported_path.unlink()

onnx_model = onnx.load(str(onnx_path))
onnx.checker.check_model(onnx_model)
print("PyTorch checkpoint：{}".format(checkpoint_path))
print("FP32 ONNX：{} ({:.2f} MiB)".format(onnx_path, onnx_path.stat().st_size / 1024 ** 2))
print("导出配置：固定 batch=1、640×640、opset=17、原始检测头（不含 NMS）")
for tensor in onnx_model.graph.input:
    dimensions = [dimension.dim_value for dimension in tensor.type.tensor_type.shape.dim]
    print("输入：{}，shape={}，elem_type={}".format(
        tensor.name, dimensions, tensor.type.tensor_type.elem_type
    ))
for tensor in onnx_model.graph.output:
    dimensions = [dimension.dim_value for dimension in tensor.type.tensor_type.shape.dim]
    print("输出：{}，shape={}，elem_type={}".format(
        tensor.name, dimensions, tensor.type.tensor_type.elem_type
    ))
