# YOLO26n-cls C++ demo 验证记录

## 环境与运行方式

在 Jetson Orin Nano Super 上分别编译运行 baseline 与 V1。两个程序加载同一个 `yolo26n-cls_fp16.engine` 和 `assets/sample.jpg`，执行 CUDA 与 OpenCV 两种预处理路径。输出时间从已解码的图像开始计量；TensorRT engine 初始化和 JPEG 解码不计入端到端时间。最近一次系统验证通过 `03-system-validation.sh` 同时保存了程序输出和 `tegrastats`。

```bash
cd models/YOLO26n-CLS/05_model_demo/baseline
cmake -S . -B build
cmake --build build -j2
./build/yolo26n_cls_demo

cd ../v1
cmake -S . -B build
cmake --build build -j2
./build/yolo26n_cls_v1_demo
```

## 实测输出

| 程序 | 预处理 | Top-1 | 预处理 (ms) | TensorRT 推理 (ms) | 端到端 (ms) |
|---|---|---|---:|---:|---:|
| Baseline | CUDA | tench (97.61%) | 0.920 | 1.292 | 1.161 |
| Baseline | OpenCV | tench (97.51%) | 2.331 | 1.196 | 3.507 |
| V1 CUDA Graph | CUDA | tench (97.61%) | 0.808 | 1.061 | 1.341 |
| V1 CUDA Graph | OpenCV | tench (97.51%) | 4.662 | 0.849 | 4.998 |

四条路径都得到与 Python / TensorRT 一致的 `tench` Top-1。CUDA 和 OpenCV 预处理的概率略有差异，但分类结果一致。推理时间是 100 次 engine 执行的平均值；每条端到端时间是单次已 warmup 请求。本表是一次短时 demo 验证，预处理与单次端到端计时波动明显，不能据此比较稳定性能或宣称 V1 的端到端加速；重复性能数据见 `../04_model_optimize/` 报告。

本次 12 个 `tegrastats` 采样点中，RAM 峰值为 `4509/7620 MiB`，GR3D 最高记录 `99%`，`VDD_IN` 最高记录 `7284 mW`，GPU 温度最高约 `49.6°C`。日志覆盖 baseline 和 V1 两次短 demo，不代表长时间温度、功耗或稳定性测试；原始记录在 `metrics/tegrastats.log`，两次程序输出在 `metrics/` 下。

演示视频为 [`demo.mp4`](demo.mp4)，由 [`04-make-demo-video.py`](04-make-demo-video.py) 根据这次 C++ 实测输出和固定样例图片生成，展示 baseline / V1 的 CUDA / OpenCV 四条分类路径。

## 交付内容

- `baseline/`：TensorRT C++ API 推理、CUDA 预处理 kernel、OpenCV CPU 预处理。
- `v1/`：固定地址下 capture / replay TensorRT enqueue 的 CUDA Graph 版本。
- 两个目录均包含运行所需的 engine、测试图片、ImageNet 类别名称表及 CMake 构建说明。
