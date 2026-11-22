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
logger = trt.Logger(trt.Logger.VERBOSE)
print("TensorRT logger：VERBOSE（显示解析、构建和 tactic 选择的详细日志）", flush=True)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)
print("解析 ONNX...", flush=True)
if not parser.parse_from_file(str(onnx_path)):
    for index in range(parser.num_errors):
        print(parser.get_error(index))
    raise RuntimeError("TensorRT 无法解析 ONNX")

print("网络层数：{}".format(network.num_layers), flush=True)
print("网络层明细：", flush=True)
for index in range(network.num_layers):
    layer = network.get_layer(index)
    layer_inputs = [
        "{} shape={} dtype={}".format(layer.get_input(input_index).name,
                                      tuple(layer.get_input(input_index).shape),
                                      layer.get_input(input_index).dtype)
        for input_index in range(layer.num_inputs)
        if layer.get_input(input_index) is not None
    ]
    layer_outputs = [
        "{} shape={} dtype={}".format(layer.get_output(output_index).name,
                                      tuple(layer.get_output(output_index).shape),
                                      layer.get_output(output_index).dtype)
        for output_index in range(layer.num_outputs)
        if layer.get_output(output_index) is not None
    ]
    print("  [{:03d}] {} | {} | 输入：{} | 输出：{}".format(
        index, layer.name, layer.type, "; ".join(layer_inputs), "; ".join(layer_outputs)
    ), flush=True)
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
print("构建配置摘要：", flush=True)
print("  精度：启用 FP16 tactic 搜索；ONNX 输入/输出 dtype 保持网络定义", flush=True)
print("  Workspace 上限：{} bytes ({:.0f} MiB)".format(workspace_size, workspace_size / 1024 ** 2), flush=True)
print("  网络层数：{}".format(network.num_layers), flush=True)
print("  输入数 / 输出数：{} / {}".format(network.num_inputs, network.num_outputs), flush=True)
print("开始构建 engine；TensorRT 将输出 tactic 选择和构建阶段日志...", flush=True)
started_at = perf_counter()
serialized_engine = builder.build_serialized_network(network, config)
build_seconds = perf_counter() - started_at
if serialized_engine is None:
    raise RuntimeError("TensorRT engine 构建失败")

engine_path.write_bytes(serialized_engine)
print("构建结果摘要：", flush=True)
print("  Engine：{}".format(engine_path.resolve()), flush=True)
print("  序列化大小：{} bytes ({:.2f} MiB)".format(
    engine_path.stat().st_size, engine_path.stat().st_size / 1024 ** 2
), flush=True)
print("  构建耗时：{:.1f} s".format(build_seconds), flush=True)
