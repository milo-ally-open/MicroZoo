#include "preprocess.h"

#include <cuda_runtime.h>

#include <cstddef>

namespace {

__global__ void letterbox_bgr_to_rgb_nchw_kernel(
    const unsigned char* source_bgr,
    float* destination_rgb_nchw,
    int source_width,
    int source_height,
    std::size_t source_stride,
    int input_size,
    int resized_width,
    int resized_height,
    int pad_left,
    int pad_top) {
    const int output_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int plane_size = input_size * input_size;
    const int output_elements = 3 * plane_size;
    if (output_index >= output_elements) {
        return;
    }

    const int channel = output_index / plane_size;
    const int pixel_index = output_index % plane_size;
    const int output_y = pixel_index / input_size;
    const int output_x = pixel_index % input_size;
    const int resized_x = output_x - pad_left;
    const int resized_y = output_y - pad_top;
    if (resized_x < 0 || resized_x >= resized_width ||
        resized_y < 0 || resized_y >= resized_height) {
        destination_rgb_nchw[output_index] = 114.0F / 255.0F;
        return;
    }

    // Match OpenCV INTER_LINEAR's half-pixel sampling and replicated borders.
    float source_x = (static_cast<float>(resized_x) + 0.5F) *
                     static_cast<float>(source_width) / resized_width - 0.5F;
    float source_y = (static_cast<float>(resized_y) + 0.5F) *
                     static_cast<float>(source_height) / resized_height - 0.5F;
    source_x = fminf(fmaxf(source_x, 0.0F), static_cast<float>(source_width - 1));
    source_y = fminf(fmaxf(source_y, 0.0F), static_cast<float>(source_height - 1));

    const int x0 = static_cast<int>(floorf(source_x));
    const int y0 = static_cast<int>(floorf(source_y));
    const int x1 = min(x0 + 1, source_width - 1);
    const int y1 = min(y0 + 1, source_height - 1);
    const float x_weight = source_x - x0;
    const float y_weight = source_y - y0;
    const int source_channel = 2 - channel;  // BGR input to RGB output.

    const unsigned char* row0 = source_bgr + static_cast<std::size_t>(y0) * source_stride;
    const unsigned char* row1 = source_bgr + static_cast<std::size_t>(y1) * source_stride;
    const float top = row0[x0 * 3 + source_channel] * (1.0F - x_weight) +
                      row0[x1 * 3 + source_channel] * x_weight;
    const float bottom = row1[x0 * 3 + source_channel] * (1.0F - x_weight) +
                         row1[x1 * 3 + source_channel] * x_weight;
    const float interpolated = top * (1.0F - y_weight) + bottom * y_weight;
    // The previous OpenCV path resized into an 8-bit image before normalization.
    destination_rgb_nchw[output_index] = __float2int_rn(interpolated) / 255.0F;
}

}  // namespace

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
    cudaStream_t stream) {
    if (source_bgr == nullptr || destination_rgb_nchw == nullptr ||
        source_width <= 0 || source_height <= 0 ||
        source_stride < static_cast<std::size_t>(source_width) * 3 ||
        input_size <= 0 || resized_width <= 0 || resized_width > input_size ||
        resized_height <= 0 || resized_height > input_size ||
        pad_left < 0 || pad_top < 0 ||
        pad_left + resized_width > input_size || pad_top + resized_height > input_size) {
        return cudaErrorInvalidValue;
    }

    constexpr int block_size = 256;
    const int element_count = 3 * input_size * input_size;
    const int block_count = (element_count + block_size - 1) / block_size;
    letterbox_bgr_to_rgb_nchw_kernel<<<block_count, block_size, 0, stream>>>(
        source_bgr, destination_rgb_nchw, source_width, source_height, source_stride,
        input_size, resized_width, resized_height, pad_left, pad_top);
    return cudaGetLastError();
}
