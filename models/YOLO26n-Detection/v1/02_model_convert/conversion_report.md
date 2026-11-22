# YOLO26n — 阶段 02 模型转换报告

## 转换配置

- 权重：`../01_model_train/checkpoints/yolo26n.pt`，Ultralytics assets release `v8.4.0` 官方 COCO 检测权重。
- Ultralytics：`8.4.155`；ONNX：opset 17。
- 输入：静态 `[1,3,640,640]` FP32，RGB NCHW，像素 `[0,1]`。
- 输出：静态 `[1,300,6]` FP32；字段顺序由 end-to-end 检测头输出，验证/演示程序进一步做阈值过滤、坐标映射和 NMS。
- 部署主路径仍使用完整 ONNX 构建 TensorRT engine；另提供 backbone+Neck 与 Detect head 的独立 ONNX 切分图，用于结构检查和切分边界验证。

## 图检查

`04-check_onnx.py` 使用 ONNX checker 检查完整导出图。完整 ONNX 有 445 个节点，TensorRT 解析为 515 层。输入名 `images`，输出名 `output0`。输出前驱包含检测头 TopK / GatherElements / Concat；图中没有 `NonMaxSuppression` 节点，因此 C++ demo 在 CPU 上进行置信度过滤与按类 NMS。可在 Netron 查看 `onnx/yolo26n.onnx`，检查 `images`、`output0` 和 `/model.23/` 检测头节点。

## Backbone/Neck 与 Detect head 切分

YOLO26n 的检测头接收三个尺度的特征图，所以不能像分类模型那样把最终 Detect 层之前的所有层简单串成 `nn.Sequential`。切分脚本会按照 Ultralytics 每层的 `f` 路由关系重放模型图，并将 Detect 实际读取的三个特征张量作为 backbone+Neck 的输出。

| 子图 | 输入 | 输出 |
|---|---|---|
| `yolo26n_backbone_neck.onnx` | `images` FP32 `[1,3,640,640]` | `p3` `[1,64,80,80]`、`p4` `[1,128,40,40]`、`p5` `[1,256,20,20]` |
| `yolo26n_detect_head.onnx` | `p3`、`p4`、`p5` | `predictions` FP32 `[1,84,8400]` 原始输出：4 个 box 通道、80 个类别通道、8400 个候选位置 |

边界来自 YOLO26n 结构：`model[0:23]` 包含骨干网络和 Neck，最终 Detect 层是 `model[23]`；Detect 从 `model[16]`、`model[19]`、`model[22]` 读取 P3/P4/P5。直接导出这个 checkpoint 的 Detect 模块时，`end2end=False`，所以它返回 `[1,84,8400]` 原始候选；这正是脚本实际得到的 shape，原来的 `[1,300,6]` 检查值不适用于这个切分点。脚本 `03-onnx_slice.py` 会比较完整 PyTorch forward 与切分后的 raw Detect 输出，并检查 P3/P4/P5 和 Detect 原始输出 shape。

完整部署 ONNX 当前为 `[1,300,6]`，其图中还包含 TopK/GatherElements 等候选筛选节点；它不是上表中的 raw Detect 切分输出。切分图用于观察 backbone/Neck 与 Detect 的边界，不能直接把 raw head 输出当成完整 ONNX 输出。当前部署仍走完整 ONNX → TensorRT 主路径。按本次“只写逻辑、不运行代码”的要求，修正后的切分脚本尚未重新运行；需在目标环境运行后记录实际误差和切分 ONNX 文件大小。

切分图用于明确模块边界和后续复用，不改变当前完整 ONNX → TensorRT engine → C++ demo 主路径；本次不单独构建 backbone/head TensorRT engines。

## TensorRT 构建

在目标 Jetson Orin Nano Super、TensorRT `10.3.0` 上解析完整 ONNX 并构建 FP16 engine。配置固定 batch 1、640×640、workspace 上限 512 MiB。构建成功，耗时约 `518.2 s`，engine 文件 `8.04 MiB`。engine 保存在 `engine/yolo26n_fp16.engine`；二进制与 Jetson TensorRT 版本/硬件绑定，需在目标设备重建。

## 复现顺序

```bash
cd models/YOLO26n-Detection/v1/02_model_convert
python3 02-check_layers.py
python3 01-export_onnx.py
python3 03-onnx_slice.py
python3 04-check_onnx.py
python3 05-build_tensorrt.py
```
