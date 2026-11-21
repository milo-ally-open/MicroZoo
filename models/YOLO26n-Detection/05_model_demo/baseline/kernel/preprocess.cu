#include "preprocess.h"

#include <cmath>

namespace {

constexpr int kInputSize = 640;

__global__ void letterbox_preprocess_kernel(
    const unsigned char* bgr_image,
    float* input,
    int source_width,
    int source_height,
    int resized_width,
    int resized_height,
    int pad_left,
    int pad_top) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr int pixel_count = kInputSize * kInputSize;
    if (index >= pixel_count) {
        return;
    }

    const int output_x = index % kInputSize;
    const int output_y = index / kInputSize;
    const int resized_x = output_x - pad_left;
    const int resized_y = output_y - pad_top;
    const float pad_value = 114.0F / 255.0F;
    float red = pad_value;
    float green = pad_value;
    float blue = pad_value;

    if (resized_x >= 0 && resized_x < resized_width && resized_y >= 0 && resized_y < resized_height) {
        const float scale_x = static_cast<float>(resized_width) / source_width;
        const float scale_y = static_cast<float>(resized_height) / source_height;
        const float source_x = fminf(fmaxf((resized_x + 0.5F) / scale_x - 0.5F, 0.0F), source_width - 1.0F);
        const float source_y = fminf(fmaxf((resized_y + 0.5F) / scale_y - 0.5F, 0.0F), source_height - 1.0F);
        const int x0 = static_cast<int>(floorf(source_x));
        const int y0 = static_cast<int>(floorf(source_y));
        const int x1 = min(x0 + 1, source_width - 1);
        const int y1 = min(y0 + 1, source_height - 1);
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
        red = r / 255.0F;
        green = g / 255.0F;
        blue = b / 255.0F;
    }

    input[index] = red;
    input[pixel_count + index] = green;
    input[2 * pixel_count + index] = blue;
}

}  // namespace

cudaError_t launch_preprocess_kernel(
    const unsigned char* device_bgr_image,
    float* device_input,
    int source_width,
    int source_height,
    int resized_width,
    int resized_height,
    int pad_left,
    int pad_top,
    cudaStream_t stream) {
    constexpr int threads = 256;
    constexpr int pixels = kInputSize * kInputSize;
    const int blocks = (pixels + threads - 1) / threads;
    letterbox_preprocess_kernel<<<blocks, threads, 0, stream>>>(
        device_bgr_image,
        device_input,
        source_width,
        source_height,
        resized_width,
        resized_height,
        pad_left,
        pad_top);
    return cudaGetLastError();
}
