# YOLO26n TensorRT C++ v1 Demo

该 Demo 从摄像头持续读取画面，并将原始 BGR 帧上传到 GPU。CUDA 预处理 kernel 在 GPU 上完成双线性缩放、RGB 转换、归一化和 NCHW 排布，再由 TensorRT FP16 engine 推理。其他 CUDA kernel 完成置信度过滤、坐标映射、两两 IoU 判断、greedy class-aware NMS 和矩形边框绘制；OpenCV 在 CPU 上叠加类别文字与 FPS。窗口每帧显示最近一秒的平均 `READ`（`camera.read` 耗时）、`SHOW`、FPS、预处理、TensorRT、CUDA 后处理和端到端时间；终端每秒打印一行相同统计。`READ` 包含 `camera.read` 等待新帧的时间。E2E 从读取摄像头帧前开始计时，到 `imshow` / `waitKey` 返回为止。

## 构建与运行

需要 JetPack 提供的 TensorRT、CUDA Toolkit/runtime、OpenCV 4（包含 highgui、videoio）、CMake 和 C++17 编译器。在本目录执行：

```bash
cmake -S . -B build
cmake --build build -j2
./build/yolo26n_demo
```

程序默认打开摄像头设备 `0`，并请求 `YUYV 640x480 @ 30 FPS`；启动信息会显示 OpenCV 是否接受设置以及报告的帧率/分辨率，首帧日志会显示实际帧尺寸。摄像头驱动可能不接受请求值，需以实际帧尺寸和 `READ` 计时为准。在检测窗口按 `q` 或 `Esc` 退出。运行时从当前目录读取 `models/yolo26n_fp16.engine` 和 `assets/coco_classes.txt`。

TensorRT engine 与构建时的 TensorRT 版本和目标设备相关。当前设备的 TensorRT 已升级到 10.7；为避免旧 engine 版本不匹配，先在目标设备重新构建：

```bash
cd ../02_model_convert
python3 05-build_tensorrt.py
mkdir -p ../05_model_demo/models
cp engine/yolo26n_fp16.engine ../05_model_demo/models/
cd ../05_model_demo
```

`CHECK_CUDA(operation)` 是一个只展开为检查函数调用的宏；`__check_cuda_runtime` 负责输出 CUDA 错误名、错误描述、文件和行号，并返回 `bool`。`enqueueV3` 使用显式创建的 CUDA stream，TensorRT、后处理和画框 kernel 在同一 stream 中按顺序执行。
