#include "postprocess.h"

#include <cuda_runtime.h>

#include <cstddef>

namespace {

constexpr int max_nms_candidates = 300;

__device__ float clamp_coordinate(float value, float upper_bound) {
    return fminf(fmaxf(value, 0.0F), upper_bound);
}

__global__ void filter_and_restore_boxes(
    const float* predictions,
    Detection* candidates,
    int* valid,
    int candidate_count,
    int class_count,
    float confidence_threshold,
    float scale,
    int pad_left,
    int pad_top,
    int image_width,
    int image_height) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= candidate_count) {
        return;
    }

    const float* prediction = predictions + index * 6;
    const float confidence = prediction[4];
    const int class_id = __float2int_rn(prediction[5]);
    if (confidence < confidence_threshold || class_id < 0 || class_id >= class_count) {
        valid[index] = 0;
        candidates[index].x1 = 0.0F;
        candidates[index].y1 = 0.0F;
        candidates[index].x2 = 0.0F;
        candidates[index].y2 = 0.0F;
        candidates[index].confidence = 0.0F;
        candidates[index].class_id = -1;
        return;
    }

    candidates[index].x1 = clamp_coordinate(
        (prediction[0] - pad_left) / scale, static_cast<float>(image_width));
    candidates[index].y1 = clamp_coordinate(
        (prediction[1] - pad_top) / scale, static_cast<float>(image_height));
    candidates[index].x2 = clamp_coordinate(
        (prediction[2] - pad_left) / scale, static_cast<float>(image_width));
    candidates[index].y2 = clamp_coordinate(
        (prediction[3] - pad_top) / scale, static_cast<float>(image_height));
    candidates[index].confidence = confidence;
    candidates[index].class_id = class_id;
    valid[index] = 1;
}

__device__ float intersection_over_union(const Detection& left, const Detection& right) {
    const float intersection_width = fmaxf(
        0.0F, fminf(left.x2, right.x2) - fmaxf(left.x1, right.x1));
    const float intersection_height = fmaxf(
        0.0F, fminf(left.y2, right.y2) - fmaxf(left.y1, right.y1));
    const float intersection = intersection_width * intersection_height;
    const float left_area = fmaxf(0.0F, left.x2 - left.x1) * fmaxf(0.0F, left.y2 - left.y1);
    const float right_area = fmaxf(0.0F, right.x2 - right.x1) * fmaxf(0.0F, right.y2 - right.y1);
    return intersection / fmaxf(left_area + right_area - intersection, 1.0e-12F);
}

__global__ void build_class_aware_suppression_mask(
    const Detection* candidates,
    const int* valid,
    unsigned char* suppression_mask,
    int candidate_count,
    float iou_threshold) {
    const int pair_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int pair_count = candidate_count * candidate_count;
    if (pair_index >= pair_count) {
        return;
    }

    suppression_mask[pair_index] = 0;
    const int current_index = pair_index / candidate_count;
    const int other_index = pair_index % candidate_count;
    if (current_index == other_index || valid[current_index] == 0 || valid[other_index] == 0) {
        return;
    }

    const Detection current = candidates[current_index];
    const Detection other = candidates[other_index];
    if (current.class_id != other.class_id) {
        return;
    }

    // 同分时由较早的网络输出候选保留，与 NMS 排序规则一致。
    if (current.confidence < other.confidence ||
        (current.confidence == other.confidence && current_index > other_index)) {
        return;
    }
    if (intersection_over_union(current, other) > iou_threshold) {
        suppression_mask[pair_index] = 1;
    }
}

__global__ void greedy_nms_kernel(
    const Detection* candidates,
    const int* valid,
    const unsigned char* suppression_mask,
    Detection* detections,
    int* detection_count,
    int candidate_count) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    int order[max_nms_candidates];
    unsigned char suppressed[max_nms_candidates];
    int valid_count = 0;
    for (int index = 0; index < candidate_count; ++index) {
        suppressed[index] = 0;
        if (valid[index] == 0) {
            continue;
        }

        // 插入排序最多处理 300 个候选，按分数降序、原始索引升序排列。
        int position = valid_count;
        while (position > 0) {
            const int previous = order[position - 1];
            const bool current_precedes_previous =
                candidates[index].confidence > candidates[previous].confidence ||
                (candidates[index].confidence == candidates[previous].confidence && index < previous);
            if (!current_precedes_previous) {
                break;
            }
            order[position] = previous;
            --position;
        }
        order[position] = index;
        ++valid_count;
    }

    int kept_count = 0;
    for (int order_index = 0; order_index < valid_count; ++order_index) {
        const int current = order[order_index];
        if (suppressed[current] != 0) {
            continue;
        }
        detections[kept_count++] = candidates[current];

        for (int next_index = order_index + 1; next_index < valid_count; ++next_index) {
            const int next = order[next_index];
            if (suppressed[next] == 0 &&
                suppression_mask[current * candidate_count + next] != 0) {
                suppressed[next] = 1;
            }
        }
    }
    *detection_count = kept_count;
}

__global__ void mark_box_edges_kernel(
    unsigned int* box_mask,
    const Detection* detections,
    const int* detection_count,
    int max_detections,
    int image_width,
    int image_height) {
    const int perimeter = 2 * image_width + 2 * image_height;
    const int position = blockIdx.x * blockDim.x + threadIdx.x;
    const int detection_index = position / perimeter;
    if (detection_index >= max_detections || detection_index >= *detection_count) {
        return;
    }

    const Detection detection = detections[detection_index];
    int left = max(0, min(image_width - 1, __float2int_rn(detection.x1)));
    int right = max(0, min(image_width - 1, __float2int_rn(detection.x2)));
    int top = max(0, min(image_height - 1, __float2int_rn(detection.y1)));
    int bottom = max(0, min(image_height - 1, __float2int_rn(detection.y2)));
    if (left > right) {
        const int temporary = left;
        left = right;
        right = temporary;
    }
    if (top > bottom) {
        const int temporary = top;
        top = bottom;
        bottom = temporary;
    }

    const int edge_position = position % perimeter;
    int x = 0;
    int y = 0;
    if (edge_position < image_width) {
        x = edge_position;
        y = top;
    } else if (edge_position < 2 * image_width) {
        x = edge_position - image_width;
        y = bottom;
    } else if (edge_position < 2 * image_width + image_height) {
        x = left;
        y = edge_position - 2 * image_width;
    } else {
        x = right;
        y = edge_position - 2 * image_width - image_height;
    }
    if (x >= left && x <= right && y >= top && y <= bottom) {
        atomicExch(&box_mask[y * image_width + x], 1U);
    }
}

__global__ void draw_box_mask_kernel(
    unsigned char* frame,
    const unsigned int* box_mask,
    int image_width,
    int image_height,
    std::size_t frame_stride) {
    const int pixel_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int pixel_count = image_width * image_height;
    if (pixel_index >= pixel_count || box_mask[pixel_index] == 0) {
        return;
    }

    const int y = pixel_index / image_width;
    const int x = pixel_index % image_width;
    unsigned char* pixel = frame + static_cast<std::size_t>(y) * frame_stride + x * 3;
    pixel[0] = 0;   // B
    pixel[1] = 255; // G
    pixel[2] = 0;   // R
}

}  // namespace

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
    cudaStream_t stream) {
    if (candidate_count <= 0 || candidate_count > max_nms_candidates ||
        class_count <= 0 || scale <= 0.0F || image_width <= 0 || image_height <= 0) {
        return cudaErrorInvalidValue;
    }

    constexpr int block_size = 256;
    const int candidate_blocks = (candidate_count + block_size - 1) / block_size;
    filter_and_restore_boxes<<<candidate_blocks, block_size, 0, stream>>>(
        predictions, candidates, valid, candidate_count, class_count,
        confidence_threshold, scale, pad_left, pad_top, image_width, image_height);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        return status;
    }

    const int pair_count = candidate_count * candidate_count;
    const int pair_blocks = (pair_count + block_size - 1) / block_size;
    build_class_aware_suppression_mask<<<pair_blocks, block_size, 0, stream>>>(
        candidates, valid, suppression_mask, candidate_count, iou_threshold);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        return status;
    }

    // 该模型最多输出 300 个框；单线程按排序后的抑制矩阵执行精确 greedy NMS。
    greedy_nms_kernel<<<1, 1, 0, stream>>>(
        candidates, valid, suppression_mask, detections, detection_count, candidate_count);
    return cudaGetLastError();
}

cudaError_t draw_boxes_cuda(
    unsigned char* frame,
    unsigned int* box_mask,
    const Detection* detections,
    const int* detection_count,
    int max_detections,
    int image_width,
    int image_height,
    std::size_t frame_stride,
    cudaStream_t stream) {
    if (max_detections <= 0 || max_detections > max_nms_candidates ||
        image_width <= 0 || image_height <= 0 || frame_stride < static_cast<std::size_t>(image_width * 3)) {
        return cudaErrorInvalidValue;
    }

    const std::size_t pixel_count = static_cast<std::size_t>(image_width) * image_height;
    cudaError_t status = cudaMemsetAsync(box_mask, 0, pixel_count * sizeof(unsigned int), stream);
    if (status != cudaSuccess) {
        return status;
    }

    constexpr int block_size = 256;
    const int perimeter = 2 * image_width + 2 * image_height;
    const int edge_count = max_detections * perimeter;
    const int edge_blocks = (edge_count + block_size - 1) / block_size;
    mark_box_edges_kernel<<<edge_blocks, block_size, 0, stream>>>(
        box_mask, detections, detection_count, max_detections, image_width, image_height);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        return status;
    }

    const int pixel_blocks = static_cast<int>((pixel_count + block_size - 1) / block_size);
    draw_box_mask_kernel<<<pixel_blocks, block_size, 0, stream>>>(
        frame, box_mask, image_width, image_height, frame_stride);
    return cudaGetLastError();
}
