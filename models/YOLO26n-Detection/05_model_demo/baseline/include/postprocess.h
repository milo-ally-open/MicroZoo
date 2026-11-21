#pragma once

#include <cuda_runtime.h>

// Decode YOLO26 end-to-end rows [x1,y1,x2,y2,score,class], threshold them,
// and map the letterboxed coordinates back to the original image.
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
    cudaStream_t stream);

// Class-aware IoU suppression. Each thread owns one keep flag and examines
// the small fixed 300-row output produced by the exported YOLO26 head.
cudaError_t launch_fast_nms_kernel(
    float* boxes,
    const int* box_count,
    int max_boxes,
    float iou_threshold,
    cudaStream_t stream);
