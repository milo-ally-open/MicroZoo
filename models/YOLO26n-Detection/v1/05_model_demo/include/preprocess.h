#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

cudaError_t preprocess_letterbox_cuda(
    const unsigned char* source_bgr,
    float* destination_rgb_nchw,
    int source_width,
    int source_height,
    std::size_t source_stride,
    int input_size,
    int resized_width,
    int resized_height,
    int pad_left,
    int pad_top,
    cudaStream_t stream);
