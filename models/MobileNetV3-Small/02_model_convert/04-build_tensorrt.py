from pathlib import Path
from time import perf_counter

import tensorrt as trt

project_dir = Path(__file__).resolve().parent
onnx_path = project_dir / "onnx" / "mobilenet_v3_small.onnx"
engine_dir = project_dir / "engine"
engine_path = engine_dir / "mobilenet_v3_small_fp16.engine"
engine_dir.mkdir(parents=True, exist_ok=True)
if not onnx_path.is_file():
    raise FileNotFoundError("先运行 01-export_onnx.py")

print("TensorRT 版本：{}".format(trt.__version__), flush=True)
print("ONNX 文件：{} ({:.2f} MiB)".format(onnx_path, onnx_path.stat().st_size / 1024 ** 2), flush=True)

logger = trt.Logger(trt.Logger.VERBOSE)
builder = trt.Builder(logger)
network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network = builder.create_network(network_flags)
parser = trt.OnnxParser(network, logger)

print("解析 ONNX...", flush=True)
if not parser.parse_from_file(str(onnx_path)):
    for index in range(parser.num_errors):
        print(parser.get_error(index))
    raise RuntimeError("TensorRT 无法解析 ONNX 模型：{}".format(onnx_path))

print("ONNX 解析完成，网络层数：{}".format(network.num_layers), flush=True)
for index in range(network.num_inputs):
    tensor = network.get_input(index)
    print("输入 {}：shape={} dtype={}".format(tensor.name, tuple(tensor.shape), tensor.dtype), flush=True)
for index in range(network.num_outputs):
    tensor = network.get_output(index)
    print("输出 {}：shape={} dtype={}".format(tensor.name, tuple(tensor.shape), tensor.dtype), flush=True)

config = builder.create_builder_config()
workspace_size = 512 << 20
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)
config.set_flag(trt.BuilderFlag.FP16)

print("构建配置：FP16，workspace 上限 {:.0f} MiB".format(workspace_size / 1024 ** 2), flush=True)
print("开始构建 engine；TensorRT 正在搜索适合本机的算子实现...", flush=True)
build_started_at = perf_counter()
serialized_engine = builder.build_serialized_network(network, config)
build_seconds = perf_counter() - build_started_at

if serialized_engine is None:
    raise RuntimeError("TensorRT engine 构建失败")

print("Engine 构建完成，用时 {:.1f} 秒".format(build_seconds), flush=True)
print("写入 engine：{}".format(engine_path), flush=True)
with open(engine_path, "wb") as engine_file:
    engine_file.write(serialized_engine)

print("Engine 已保存：{} ({:.2f} MiB)".format(engine_path, engine_path.stat().st_size / 1024 ** 2), flush=True)
