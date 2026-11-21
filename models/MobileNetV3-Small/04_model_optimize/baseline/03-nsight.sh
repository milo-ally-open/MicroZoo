#!/usr/bin/env bash
# Run from this directory: sudo -E bash 03-nsight.sh

set -euo pipefail

ENGINE_PATH="engine/mobilenet_v3_small_fp16.engine"
PROFILE_DIR="profiling/nsys"

mkdir -p "${PROFILE_DIR}"

if ! command -v nsys >/dev/null 2>&1; then
    echo "nsys is not in PATH; install Nsight Systems CLI on the Jetson first." >&2
    exit 1
fi

if command -v trtexec >/dev/null 2>&1; then
    TRTEXEC="$(command -v trtexec)"
elif [ -x /usr/src/tensorrt/bin/trtexec ]; then
    TRTEXEC="/usr/src/tensorrt/bin/trtexec"
else
    echo "trtexec was not found in PATH or /usr/src/tensorrt/bin/." >&2
    exit 1
fi

if [ ! -f "${ENGINE_PATH}" ]; then
    echo "TensorRT engine not found: ${ENGINE_PATH}" >&2
    exit 1
fi

echo "Collecting TensorRT eager timeline..."
nsys profile \
    --trace=cuda,nvtx,osrt \
    --stats=true \
    --force-overwrite=true \
    -o "${PROFILE_DIR}/trt-eager" \
    "${TRTEXEC}" \
    --loadEngine="${ENGINE_PATH}" \
    --noDataTransfers \
    --warmUp=500 \
    --duration=3

echo "Collecting TensorRT CUDA Graph timeline..."
nsys profile \
    --trace=cuda,nvtx,osrt \
    --cuda-graph-trace=node \
    --stats=true \
    --force-overwrite=true \
    -o "${PROFILE_DIR}/trt-cudagraph" \
    "${TRTEXEC}" \
    --loadEngine="${ENGINE_PATH}" \
    --noDataTransfers \
    --useCudaGraph \
    --warmUp=500 \
    --duration=3

echo "Nsight Systems reports saved under ${PROFILE_DIR}/"
