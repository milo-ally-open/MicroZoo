from collections import Counter
from pathlib import Path

import onnx

project_dir = Path(__file__).resolve().parent
onnx_path = project_dir / "onnx" / "yolo26n.onnx"
if not onnx_path.is_file():
    raise FileNotFoundError("先运行 01-export_onnx.py")

model = onnx.load(str(onnx_path))
onnx.checker.check_model(model)
operations = Counter(node.op_type for node in model.graph.node)
producers = {output: node for node in model.graph.node for output in node.output}
print("ONNX：{}".format(onnx_path))
print("节点数：{}".format(len(model.graph.node)))
print("算子统计：{}".format(dict(sorted(operations.items()))))
for tensor in list(model.graph.input) + list(model.graph.output):
    dimensions = []
    for dimension in tensor.type.tensor_type.shape.dim:
        dimensions.append(dimension.dim_value if dimension.dim_value else dimension.dim_param)
    kind = "输入" if tensor in model.graph.input else "输出"
    print("{} {} shape={} elem_type={}".format(
        kind,
        tensor.name,
        dimensions,
        tensor.type.tensor_type.elem_type,
    ))

for tensor in model.graph.output:
    print("输出生产节点：")
    node = producers.get(tensor.name)
    if node is not None:
        print("  {} ({})".format(node.name or "<unnamed>", node.op_type))
        for input_name in node.input:
            parent = producers.get(input_name)
            if parent is not None:
                print("    <- {} ({})".format(parent.name or "<unnamed>", parent.op_type))
print("末尾 20 个节点：")
for node in model.graph.node[-20:]:
    print("  {:<24} {:<20} {} -> {}".format(
        node.name or "<unnamed>",
        node.op_type,
        ", ".join(node.input[:3]),
        ", ".join(node.output[:3]),
    ))
