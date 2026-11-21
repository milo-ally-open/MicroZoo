#pragma once

#include <cuda_runtime_api.h>

cudaError_t launch_preprocess_kernel(
    const unsigned char* device_bgr_image,
    float* device_input,
    int source_width,
    int source_height,
    int resized_width,
    int resized_height,
    int crop_left,
    int crop_top,
    cudaStream_t stream
);
