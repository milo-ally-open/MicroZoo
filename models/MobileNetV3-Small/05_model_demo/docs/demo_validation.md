# C++ Demo Smoke Validation

## Build and run

- Device: Jetson Orin Nano Super.
- Toolchain: GCC 11.4, CUDA 12.6, TensorRT 10.3, OpenCV 4.10.
- Build: `./scripts/build.sh` completed successfully after extracting the TensorRT logger to a header.
- Run: `./build/mobilenet_demo` with the bundled `assets/dog.jpg` and engine.
- Engine SHA-256 matches the stage 02 engine and the stage 04 baseline engine.

## Result

Top-1 was **Samoyed**, matching the expected sample label and the stage 03 TensorRT validation. Top-5 was Samoyed, Pomeranian, Arctic fox, wallaby, and Shetland sheepdog; stage 03 listed Great Pyrenees as fifth. The top-1 result agrees. The fifth-class difference may be related to using OpenCV image decoding/resizing instead of the Python/Pillow preprocessing path; this smoke run is not a replacement for the numerical validation in stage 03.

One request measured after engine and CUDA Graph initialization:

| Part | Time (ms) |
|---|---:|
| Image load | 29.26 |
| Preprocess | 2.13 |
| H2D + CUDA Graph inference + D2H | 1.67 |
| Top-5 postprocess | 0.03 |
| Request total | 33.10 |

Graph/engine setup took 179.45 ms and is reported separately. This is one smoke measurement, not a latency benchmark or P50/P95 result; image-load time can vary with filesystem caching. The request timing includes image decoding and is therefore not directly comparable to the V1 inference-only timing.
