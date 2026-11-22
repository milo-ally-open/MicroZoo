#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

struct Detection {
    float x1;
    float y1;
    float x2;
    float y2;
    float confidence;
    int class_id;
};

cudaError_t postprocess_cuda(
    const float* predictions,
    Detection* candidates,
    int* valid,
    unsigned char* suppression_mask,
    Detection* detections,
    int* detection_count,
    int candidate_count,
    int class_count,
    float confidence_threshold,
    float iou_threshold,
    float scale,
    int pad_left,
    int pad_top,
    int image_width,
    int image_height,
    cudaStream_t stream);

cudaError_t draw_boxes_cuda(
    unsigned char* frame,
    unsigned int* box_mask,
    const Detection* detections,
    const int* detection_count,
    int max_detections,
    int image_width,
    int image_height,
    std::size_t frame_stride,
    cudaStream_t stream);
