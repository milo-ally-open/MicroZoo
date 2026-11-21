from pathlib import Path
from time import perf_counter

import tensorrt as trt

project_dir = Path(__file__).resolve().parent
onnx_path = project_dir / "onnx" / "yolo26n-cls.onnx"
engine_dir = project_dir / "engine"
engine_path = engine_dir / "yolo26n-cls_fp16.engine"
engine_dir.mkdir(parents=True, exist_ok=True)
if not onnx_path.is_file():
    raise FileNotFoundError("先运行 01-export_onnx.py")

print("TensorRT：{}".format(trt.__version__), flush=True)
print("ONNX：{} ({:.2f} MiB)".format(onnx_path, onnx_path.stat().st_size / 1024 ** 2), flush=True)
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)
print("解析 ONNX...", flush=True)
if not parser.parse_from_file(str(onnx_path)):
    for index in range(parser.num_errors):
        print(parser.get_error(index))
    raise RuntimeError("TensorRT 无法解析 ONNX")

print("网络层数：{}".format(network.num_layers), flush=True)
for index in range(network.num_inputs):
    tensor = network.get_input(index)
    print("输入 {}：shape={} dtype={}".format(tensor.name, tuple(tensor.shape), tensor.dtype), flush=True)
for index in range(network.num_outputs):
    tensor = network.get_output(index)
    print("输出 {}：shape={} dtype={}".format(tensor.name, tuple(tensor.shape), tensor.dtype), flush=True)

config = builder.create_builder_config()
workspace_size = 512 << 20
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)
if not builder.platform_has_fast_fp16:
    print("警告：设备未报告快速 FP16 支持，仍尝试构建 FP16 engine", flush=True)
config.set_flag(trt.BuilderFlag.FP16)
print("构建配置：固定 batch=1、224×224、FP16、workspace 上限 512 MiB", flush=True)
print("开始构建 engine...", flush=True)
started_at = perf_counter()
serialized_engine = builder.build_serialized_network(network, config)
build_seconds = perf_counter() - started_at
if serialized_engine is None:
    raise RuntimeError("TensorRT engine 构建失败")

engine_path.write_bytes(serialized_engine)
print("Engine：{}".format(engine_path), flush=True)
print("构建用时：{:.1f} s".format(build_seconds), flush=True)
print("文件大小：{:.2f} MiB".format(engine_path.stat().st_size / 1024 ** 2), flush=True)
