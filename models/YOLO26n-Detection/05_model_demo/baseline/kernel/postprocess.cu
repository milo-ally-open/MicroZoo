#include "postprocess.h"

#include <cmath>

namespace {

constexpr int kBoxElements = 7;

__global__ void decode_filter_kernel(
    const float* predictions,
    int prediction_count,
    float confidence_threshold,
    float scale,
    int pad_left,
    int pad_top,
    int original_width,
    int original_height,
    float* boxes,
    int* box_count,
    int max_boxes) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= prediction_count) {
        return;
    }

    const float* prediction = predictions + index * 6;
    const float confidence = prediction[4];
    const int class_id = static_cast<int>(roundf(prediction[5]));
    if (confidence < confidence_threshold || class_id < 0 || class_id >= 80) {
        return;
    }

    const int output_index = atomicAdd(box_count, 1);
    if (output_index >= max_boxes) {
        return;
    }
    float* box = boxes + output_index * kBoxElements;
    box[0] = fminf(fmaxf((prediction[0] - pad_left) / scale, 0.0F), static_cast<float>(original_width));
    box[1] = fminf(fmaxf((prediction[1] - pad_top) / scale, 0.0F), static_cast<float>(original_height));
    box[2] = fminf(fmaxf((prediction[2] - pad_left) / scale, 0.0F), static_cast<float>(original_width));
    box[3] = fminf(fmaxf((prediction[3] - pad_top) / scale, 0.0F), static_cast<float>(original_height));
    box[4] = confidence;
    box[5] = static_cast<float>(class_id);
    box[6] = 1.0F;
}

__device__ float box_iou(const float* a, const float* b) {
    const float left = fmaxf(a[0], b[0]);
    const float top = fmaxf(a[1], b[1]);
    const float right = fminf(a[2], b[2]);
    const float bottom = fminf(a[3], b[3]);
    const float intersection = fmaxf(right - left, 0.0F) * fmaxf(bottom - top, 0.0F);
    if (intersection == 0.0F) {
        return 0.0F;
    }
    const float area_a = fmaxf(a[2] - a[0], 0.0F) * fmaxf(a[3] - a[1], 0.0F);
    const float area_b = fmaxf(b[2] - b[0], 0.0F) * fmaxf(b[3] - b[1], 0.0F);
    return intersection / fmaxf(area_a + area_b - intersection, 1.0e-12F);
}

__global__ void fast_nms_kernel(float* boxes, const int* box_count, int max_boxes, float iou_threshold) {
    const int position = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = min(*box_count, max_boxes);
    if (position >= count) {
        return;
    }

    float* current = boxes + position * kBoxElements;
    for (int index = 0; index < count; ++index) {
        const float* candidate = boxes + index * kBoxElements;
        if (index == position || current[5] != candidate[5]) {
            continue;
        }
        if (candidate[4] > current[4] || (candidate[4] == current[4] && index < position)) {
            if (box_iou(current, candidate) > iou_threshold) {
                current[6] = 0.0F;
                return;
            }
        }
    }
}

}  // namespace

cudaError_t launch_decode_filter_kernel(
    const float* predictions,
    int prediction_count,
    float confidence_threshold,
    float scale,
    int pad_left,
    int pad_top,
    int original_width,
    int original_height,
    float* boxes,
    int* box_count,
    int max_boxes,
    cudaStream_t stream) {
    constexpr int threads = 128;
    const int blocks = (prediction_count + threads - 1) / threads;
    decode_filter_kernel<<<blocks, threads, 0, stream>>>(
        predictions,
        prediction_count,
        confidence_threshold,
        scale,
        pad_left,
        pad_top,
        original_width,
        original_height,
        boxes,
        box_count,
        max_boxes);
    return cudaGetLastError();
}

cudaError_t launch_fast_nms_kernel(
    float* boxes,
    const int* box_count,
    int max_boxes,
    float iou_threshold,
    cudaStream_t stream) {
    constexpr int threads = 128;
    const int blocks = (max_boxes + threads - 1) / threads;
    fast_nms_kernel<<<blocks, threads, 0, stream>>>(boxes, box_count, max_boxes, iou_threshold);
    return cudaGetLastError();
}
