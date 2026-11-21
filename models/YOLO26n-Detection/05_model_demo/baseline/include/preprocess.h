#pragma once

#include <cuda_runtime.h>

// Upload a decoded OpenCV BGR image separately, then letterbox, convert to RGB,
// normalize to [0,1], and write contiguous FP32 NCHW directly into TensorRT input.
cudaError_t launch_preprocess_kernel(
    const unsigned char* device_bgr_image,
    float* device_input,
    int source_width,
    int source_height,
    int resized_width,
    int resized_height,
    int pad_left,
    int pad_top,
    cudaStream_t stream);
