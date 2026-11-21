#include "preprocess.h"

#include <cuda_runtime.h>

__global__ void preprocess_kernel(
    const unsigned char* bgr_image,
    float* input,
    int source_width,
    int source_height,
    int resized_width,
    int resized_height,
    int crop_left,
    int crop_top
) {
    constexpr int crop_size = 224;
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= crop_size * crop_size) {
        return;
    }

    const int x = index % crop_size + crop_left;
    const int y = index / crop_size + crop_top;
    const float source_x = fminf(fmaxf(
        (static_cast<float>(x) + 0.5F) * source_width / resized_width - 0.5F,
        0.0F
    ), static_cast<float>(source_width - 1));
    const float source_y = fminf(fmaxf(
        (static_cast<float>(y) + 0.5F) * source_height / resized_height - 0.5F,
        0.0F
    ), static_cast<float>(source_height - 1));
    const int x0 = static_cast<int>(floorf(source_x));
    const int y0 = static_cast<int>(floorf(source_y));
    const int x1 = x0 + 1 < source_width ? x0 + 1 : x0;
    const int y1 = y0 + 1 < source_height ? y0 + 1 : y0;
    const float dx = source_x - x0;
    const float dy = source_y - y0;

    const unsigned char* p00 = bgr_image + (y0 * source_width + x0) * 3;
    const unsigned char* p01 = bgr_image + (y0 * source_width + x1) * 3;
    const unsigned char* p10 = bgr_image + (y1 * source_width + x0) * 3;
    const unsigned char* p11 = bgr_image + (y1 * source_width + x1) * 3;
    const float b = (p00[0] * (1.0F - dx) + p01[0] * dx) * (1.0F - dy) +
                    (p10[0] * (1.0F - dx) + p11[0] * dx) * dy;
    const float g = (p00[1] * (1.0F - dx) + p01[1] * dx) * (1.0F - dy) +
                    (p10[1] * (1.0F - dx) + p11[1] * dx) * dy;
    const float r = (p00[2] * (1.0F - dx) + p01[2] * dx) * (1.0F - dy) +
                    (p10[2] * (1.0F - dx) + p11[2] * dx) * dy;
    constexpr int plane_size = crop_size * crop_size;
    input[index] = (floorf(r + 0.5F) / 255.0F - 0.485F) / 0.229F;
    input[plane_size + index] = (floorf(g + 0.5F) / 255.0F - 0.456F) / 0.224F;
    input[2 * plane_size + index] = (floorf(b + 0.5F) / 255.0F - 0.406F) / 0.225F;
}

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
) {
    constexpr int crop_size = 224;
    constexpr int threads = 256;
    constexpr int pixels = crop_size * crop_size;
    const int blocks = (pixels + threads - 1) / threads;
    preprocess_kernel<<<blocks, threads, 0, stream>>>(
        device_bgr_image,
        device_input,
        source_width,
        source_height,
        resized_width,
        resized_height,
        crop_left,
        crop_top
    );
    return cudaGetLastError();
}
