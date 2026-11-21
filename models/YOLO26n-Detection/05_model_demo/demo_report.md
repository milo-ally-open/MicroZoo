# YOLO26n — 阶段 05 实时摄像头演示报告

## 应用链路

`baseline/src/main.cpp` 打开 `/dev/video0`（V4L2，摄像头返回 1280×720），从每帧 BGR 输入开始执行：CUDA letterbox / RGB-NCHW 归一化 → TensorRT FP16 engine → CUDA 置信度过滤 / 坐标映射 / class-aware NMS → 回传检测框 → OpenCV 绘制和显示。窗口按 `q` 或 Esc 退出。主程序不依赖单张测试图片。

## 构建与现场验证

在 Jetson Orin Nano Super、JetPack R36.4.3、TensorRT 10.3.0、CUDA 12.6、OpenCV 4.10.0 上，`cmake -S . -B build` 和 `cmake --build build -j2` 构建成功。摄像头实时运行通过第 30 帧；该帧画面显示 4 个检测框。终端采样：预处理约 `1.65 ms`、TensorRT 推理约 `20.22 ms`、CUDA 后处理及结果回传约 `0.22 ms`、端到端处理约 `22.09 ms`，处理吞吐约 `45.3 frame/s`。

该单次摄像头运行用于验证采集—推理—后处理—显示完整链路，不是稳定的性能基准；处理吞吐计算不包含摄像头采集等待和 GUI 绘制，也不代表摄像头的实际采集帧率。正式性能数据见阶段 04 多轮基准。

## 复现

```bash
cd models/YOLO26n-Detection/05_model_demo/baseline
python3 00-prepare_assets.py
cmake -S . -B build
cmake --build build -j2
./build/yolo26n_demo
```

演示需要摄像头 `/dev/video0` 和可用的桌面显示。若摄像头使用其他设备索引，修改 `main.cpp` 中 `VideoCapture` 的索引。engine 在当前 Jetson 上构建，其他 Jetson 或 TensorRT 版本需从 ONNX 重新构建。
