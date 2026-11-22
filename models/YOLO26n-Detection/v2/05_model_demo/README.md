# YOLO26n TensorRT INT8 实时摄像头 Demo

该 Demo 沿用 v1 的摄像头模式，并用 GStreamer 固定设置 UVC 摄像头为 YUYV/YUY2 640×480、30 FPS，避免由 OpenCV/V4L2 自动协商到较低帧率模式。板上 `nvvidconv compute-hw=2` 使用 VIC 将 YUY2 转成 BGRx，`videoconvert` 去掉 alpha 后由 appsink 输出 BGR 给 OpenCV；appsink 限制为一个 buffer、丢弃旧帧并关闭时钟同步，避免处理落后时排队显示旧画面。此设备上的 `/dev/video0` 是 USB 摄像头，所以采用 `v4l2src`；CSI 摄像头才使用 Argus 的 `nvarguscamerasrc`。

可先单独验证采集格式和 VIC 转换是否能协商：

```bash
gst-launch-1.0 -v v4l2src device=/dev/video0 io-mode=2 num-buffers=5 \
  ! 'video/x-raw,format=YUY2,width=640,height=480,framerate=30/1' \
  ! nvvidconv compute-hw=2 ! 'video/x-raw,format=BGRx' \
  ! videoconvert ! 'video/x-raw,format=BGR' ! fakesink sync=false
```

OpenCV 从 GStreamer appsink 取得 BGR 帧并上传到 GPU；`kernel/preprocess.cu` 在 CUDA 上完成等比例缩放、114 灰边 LetterBox、双线性插值、BGR→RGB、`[0,1]` 归一化和 NCHW 排布；TensorRT 执行剪枝后的 INT8 engine；CUDA kernel 完成置信度过滤、坐标还原、class-aware greedy NMS 和矩形框绘制。CPU 负责类别文字/FPS 叠加和窗口显示。

摄像头模式固定为与 v1 一致的 640×480@30；启动时会打印 GStreamer pipeline 和协商后的帧率/分辨率。GStreamer 完成摄像头采集与 VIC 色彩转换，后续预处理仍沿用 v1 CUDA kernel。CUDA 预处理使用相同半像素双线性采样坐标，并把插值结果舍入到 8-bit 后归一化。输入 tensor 仍为 FP32 `[1,3,640,640]`；INT8 是 TensorRT engine 内部可用的量化执行精度。摄像头帧的 H2D 复制在预处理 kernel 前、同一 CUDA stream 中排队；画面中的 `PRE` 时间只计 CUDA 预处理 kernel，不包含 H2D。`READ` 是从 `camera.read()` 开始到 appsink 返回 BGR 帧的时间，包含 GStreamer 采集和颜色转换；E2E 从开始读取摄像头帧计到 `imshow`/`waitKey` 返回；FPS 反映完整摄像头处理循环。

## 构建与运行

需要 JetPack 提供的 TensorRT、CUDA Toolkit/runtime、OpenCV 4（包含 highgui、videoio 和 GStreamer backend）、GStreamer 的 `v4l2src`/`nvvidconv`/`videoconvert`/`appsink` 插件、CMake 和 C++17 编译器。在目标 Jetson 上构建 engine 并复制到本目录：

```bash
cd ../02_model_convert
python3 04-build-tensorrt-int8.py
mkdir -p ../05_model_demo/models
cp engine/yolo26n_pruned_int8.engine ../05_model_demo/models/
cd ../05_model_demo
cmake -S . -B build
cmake --build build -j2
./build/yolo26n_int8_demo
```

程序固定打开 `/dev/video0`，请求 YUY2 640×480@30。窗口和终端每秒显示 FPS、CUDA preprocess、TensorRT inference、CUDA postprocess、READ、显示和端到端时间；按 `q` 或 `Esc` 退出。GStreamer pipeline 可在源码顶部的 `camera_pipeline` 常量中调整。

`CHECK_CUDA(operation)` 是一个只转发到检查函数的宏；`__check_cuda_runtime` 负责输出 CUDA 错误名、错误描述、文件和行号，并返回 `bool`。TensorRT、预处理 kernel、推理、NMS 和画框都使用显式 CUDA stream，并在同一 stream 中顺序执行。

在当前板上用 `gst-launch-1.0` 对相同的 V4L2→VIC→BGR pipeline 和 640×480@30 模式做了 5 帧采集检查，caps 协商成功；OpenCV 构建启用了 GStreamer backend，C++ demo 也已 clean build。OpenCV 直接读帧检查时 `/dev/video0` 正被 `yolo26n_demo` 占用，因此没有启动第二个摄像头客户端；关闭 v1 demo 后再运行 v2。
