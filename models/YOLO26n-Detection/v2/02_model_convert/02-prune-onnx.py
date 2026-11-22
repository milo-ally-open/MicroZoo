from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

project_dir = Path(__file__).resolve().parent
source_path = project_dir / "onnx" / "yolo26n_fp32.onnx"
output_path = project_dir / "onnx" / "yolo26n_pruned.onnx"
target_node_name = "/model.7/conv/Conv"
keep_ratio = 0.98

if not source_path.is_file():
    raise FileNotFoundError("先运行 01_export_onnx.py：{}".format(source_path))

model = onnx.load(str(source_path))
initializers = {tensor.name: tensor for tensor in model.graph.initializer}
consumers = {}
# ONNX 把计算图表示为 tensor 边。先按每个 tensor 建立消费者索引，
# 这样可以确认目标通道是否还被其他分支使用，避免只裁一层而破坏图结构。
for node in model.graph.node:
    for input_name in node.input:
        consumers.setdefault(input_name, []).append(node)

target_nodes = [node for node in model.graph.node if node.name == target_node_name]
if len(target_nodes) != 1 or target_nodes[0].op_type != "Conv":
    raise RuntimeError("找不到唯一剪枝目标 Conv：{}".format(target_node_name))
target_node = target_nodes[0]
target_attributes = {attribute.name: onnx.helper.get_attribute_value(attribute) for attribute in target_node.attribute}
if target_attributes.get("group", 1) != 1:
    raise RuntimeError("目标 Conv 不是普通卷积，不能按此脚本裁剪")
if len(target_node.input) < 2 or target_node.input[1] not in initializers:
    raise RuntimeError("目标 Conv 权重不是 ONNX initializer")

target_weight_tensor = initializers[target_node.input[1]]
target_weight = numpy_helper.to_array(target_weight_tensor)
if target_weight.ndim != 4 or target_weight.shape[0] != 256:
    raise RuntimeError("目标 Conv 权重 shape 变化：{}".format(target_weight.shape))
target_output_channels = target_weight.shape[0]
direct_consumers = consumers.get(target_node.output[0], [])
# 本模型的目标卷积后接 SiLU：Sigmoid(x) 与 x 做 Mul。
# 两条分支必须同步保留同一组通道，后续 Conv 的输入权重也必须同步切片。
sigmoid_nodes = [node for node in direct_consumers if node.op_type == "Sigmoid"]
mul_nodes = [node for node in direct_consumers if node.op_type == "Mul"]
if len(direct_consumers) != 2 or len(sigmoid_nodes) != 1 or len(mul_nodes) != 1:
    raise RuntimeError("预期目标 Conv 输出只连接 Sigmoid 和 Mul，实际为：{}".format(
        [node.op_type for node in direct_consumers]
    ))

sigmoid_node = sigmoid_nodes[0]
mul_node = mul_nodes[0]
if set(mul_node.input) != {target_node.output[0], sigmoid_node.output[0]}:
    raise RuntimeError("目标节点不符合 Conv→Sigmoid/Mul 的 SiLU 连接结构")
mul_consumers = consumers.get(mul_node.output[0], [])
if len(mul_consumers) != 1 or mul_consumers[0].op_type != "Conv":
    raise RuntimeError("SiLU 输出必须只连接一个后继 Conv，实际不满足")

next_node = mul_consumers[0]
next_attributes = {attribute.name: onnx.helper.get_attribute_value(attribute) for attribute in next_node.attribute}
if next_attributes.get("group", 1) != 1 or len(next_node.input) < 2 or next_node.input[1] not in initializers:
    raise RuntimeError("后继节点不是可安全裁剪输入通道的普通 Conv")
if next_node.input[0] != mul_node.output[0]:
    raise RuntimeError("后继 Conv 输入关系变化，停止剪枝")

next_weight_tensor = initializers[next_node.input[1]]
next_weight = numpy_helper.to_array(next_weight_tensor)
if next_weight.ndim != 4 or next_weight.shape[1] != target_output_channels:
    raise RuntimeError("后继 Conv 输入通道与目标输出通道不匹配：{} vs {}".format(
        next_weight.shape, target_output_channels
    ))

# 每个输出滤波器的 L1 范数作为简单重要性分数；保留分数最大的通道。
# 对索引排序是为了保留原网络中的通道顺序，便于一致地切片后继卷积权重。
keep_count = int(target_output_channels * keep_ratio)
if keep_count <= 0 or keep_count >= target_output_channels:
    raise RuntimeError("无效保留通道数：{}".format(keep_count))
importance = np.abs(target_weight).sum(axis=(1, 2, 3))
keep_indices = np.sort(np.argsort(-importance, kind="stable")[:keep_count])
pruned_target_weight = target_weight[keep_indices, :, :, :].copy()
pruned_next_weight = next_weight[:, keep_indices, :, :].copy()

# 结构化通道裁剪必须同时改变目标 Conv 输出维与后继 Conv 输入维，
# 否则权重张量形状和中间 activation 的通道数就会对不上。
target_weight_tensor.CopyFrom(numpy_helper.from_array(pruned_target_weight, name=target_weight_tensor.name))
next_weight_tensor.CopyFrom(numpy_helper.from_array(pruned_next_weight, name=next_weight_tensor.name))
if len(target_node.input) > 2 and target_node.input[2] in initializers:
    target_bias_tensor = initializers[target_node.input[2]]
    target_bias = numpy_helper.to_array(target_bias_tensor)
    if target_bias.shape != (target_output_channels,):
        raise RuntimeError("目标 Conv bias shape 变化：{}".format(target_bias.shape))
    target_bias_tensor.CopyFrom(numpy_helper.from_array(
        target_bias[keep_indices].copy(), name=target_bias_tensor.name
    ))

# ONNX checker 检查图是否合法；shape inference 再推导裁剪后的 tensor 形状。
onnx.checker.check_model(model)
model = onnx.shape_inference.infer_shapes(model)
onnx.checker.check_model(model)
output_path.parent.mkdir(parents=True, exist_ok=True)
onnx.save(model, str(output_path))

# 剪枝后网络变了，旧网络生成的 TensorRT 校准表不能复用。
calibration_cache_path = project_dir / "calibration" / "yolo26n_int8_entropy.cache"
if calibration_cache_path.is_file():
    calibration_cache_path.unlink()
    print("已删除旧校准缓存：{}".format(calibration_cache_path))

print("剪枝目标：{}".format(target_node.name))
print("后继 Conv：{}".format(next_node.name))
print("输出通道：{} -> {}（移除 {:.1f}%）".format(target_output_channels, keep_count, (target_output_channels - keep_count) / target_output_channels * 100))
print("目标权重：{} -> {}".format(target_weight.shape, pruned_target_weight.shape))
print("后继输入权重：{} -> {}".format(next_weight.shape, pruned_next_weight.shape))
print("结构化剪枝 ONNX：{} ({:.2f} MiB)".format(output_path, output_path.stat().st_size / 1024 ** 2))
