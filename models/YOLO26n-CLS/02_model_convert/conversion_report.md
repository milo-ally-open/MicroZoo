# YOLO26n-cls 转换报告

## 转换摘要

- 状态：完整 ONNX、backbone ONNX、classification head ONNX 和完整 TensorRT engine 均已生成并验证加载。
- 转换链路：Ultralytics 官方 ImageNet-1K checkpoint → ONNX opset 17 → Jetson TensorRT FP16 engine。
- 目标：Jetson Orin Nano Super，Jetson Linux R36.4.3，CUDA 12.6.68，TensorRT 10.3.0。
- Ultralytics：8.4.155；PyTorch：2.5.0a0+872d972e41.nv24.08。
- checkpoint：`pt/yolo26n-cls.pt`，5,786,434 bytes（5.52 MiB）。

## 完整模型接口

| 产物 | 输入 | 输出 | 备注 |
|---|---|---|---|
| Full ONNX | `images` FP32 NCHW `[1,3,224,224]` | `output0` FP32 `[1,1000]` | 分类概率；图外预处理将 RGB 像素缩放到 `[0,1]` |
| TensorRT engine | `images` FP32 `[1,3,224,224]` | `output0` FP32 `[1,1000]` | Builder 启用 FP16；I/O 保持 FP32 |

Full ONNX 大小为 11,275,222 bytes（10.75 MiB），由 Ultralytics 8.4.155 导出，opset 17，固定 batch 1 / 224×224。Engine 使用 TensorRT ONNX parser，workspace 上限 512 MiB，在目标 Jetson 上构建；首次实测构建耗时约 336.6 s，engine 为 7,150,500 bytes（6.82 MiB）。TensorRT engine 与设备及软件版本相关，应在目标系统按本脚本重新构建。

## Backbone / classification head 切分

`03-onnx_slice.py` 根据已加载网络结构将 `model[0:10]` 导出为 backbone，把 `model[10]` 的 `Classify` 模块导出为 head。分类头在 eval 模式下可能返回 `(probabilities, logits)`；导出包装会统一选取概率张量，因此 head ONNX 只有一个 `[1,1000]` 输出。层检查和 Netron 的切分边界为：

| 子图 | 输入 | 输出 |
|---|---|---|
| `yolo26n-cls_backbone.onnx` | `images` `[1,3,224,224]` | `features` `[1,256,7,7]` |
| `yolo26n-cls_head.onnx` | `features` `[1,256,7,7]` | `probabilities` `[1,1000]` |

两段 PyTorch 输出最大误差为 0；ONNX Runtime 串接后与完整 ONNX 的 MAE 为 `4.36e-10`，最大绝对误差 `2.38e-7`，Top-1 一致。完整 engine 仍由完整 ONNX 构建；拆分图用于结构检查、模块化复用和切分正确性验证。

要检查图结构，可在 [Netron](https://netron.app) 分别打开 `onnx/yolo26n-cls.onnx`、`onnx/yolo26n-cls_backbone.onnx` 和 `onnx/yolo26n-cls_head.onnx`，确认 backbone 的 `[1,256,7,7]` 输出与 head 输入相连。

## 复现命令

```bash
cd models/YOLO26n-CLS/02_model_convert
python3 01-export_onnx.py
python3 02-check_layers.py
python3 03-onnx_slice.py
python3 04-build_tensorrt.py
```
