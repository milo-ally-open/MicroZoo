# YOLO26n — 阶段 02 模型转换报告

## 转换配置

- 权重：`01_model_train/checkpoints/yolo26n.pt`，Ultralytics assets release `v8.4.0` 官方 COCO 检测权重。
- Ultralytics：`8.4.155`；ONNX：opset 17。
- 输入：静态 `[1,3,640,640]` FP32，RGB NCHW，像素 `[0,1]`。
- 输出：静态 `[1,300,6]` FP32；字段顺序由 end-to-end 检测头输出，验证/演示程序进一步做阈值过滤、坐标映射和 NMS。
- 未切分网络：完整图已经可导出、构建和运行；当前没有按 backbone/head 拆分的部署需求。

## 图检查

`03-check_onnx.py` 已用 ONNX checker 检查导出图。完整 ONNX 有 445 个节点，TensorRT 解析为 515 层。输入名 `images`，输出名 `output0`。输出前驱包含检测头 TopK / GatherElements / Concat；图中没有 `NonMaxSuppression` 节点，因此 C++ demo 另有 CUDA 置信度过滤与按类 NMS。可在 Netron 查看 `onnx/yolo26n.onnx`，检查 `images`、`output0` 和 `/model.23/` 检测头节点。

## TensorRT 构建

在目标 Jetson Orin Nano Super、TensorRT `10.3.0` 上解析完整 ONNX 并构建 FP16 engine。配置固定 batch 1、640×640、workspace 上限 512 MiB。构建成功，耗时约 `518.2 s`，engine 文件 `8.04 MiB`。engine 保存在 `engine/yolo26n_fp16.engine`；二进制与 Jetson TensorRT 版本/硬件绑定，需在目标设备重建。

## 复现顺序

```bash
cd models/YOLO26n-Detection/02_model_convert
python3 02-check_layers.py
python3 01-export_onnx.py
python3 03-check_onnx.py
python3 04-build_tensorrt.py
```
