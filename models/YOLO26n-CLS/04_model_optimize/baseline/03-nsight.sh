#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE="${SCRIPT_DIR}/engine/yolo26n-cls_fp16.engine"
PROFILE_DIR="${SCRIPT_DIR}/profiling"
NSYS_DIR="${PROFILE_DIR}/nsys"
mkdir -p "${NSYS_DIR}"

if ! command -v nsys >/dev/null 2>&1; then
    echo "nsys is required; install Nsight Systems on the Jetson first." >&2
    exit 1
fi
TRTEXEC="$(command -v trtexec || true)"
if [[ -z "${TRTEXEC}" ]]; then
    for candidate in /usr/src/tensorrt/bin/trtexec /usr/bin/trtexec; do
        if [[ -x "${candidate}" ]]; then
            TRTEXEC="${candidate}"
            break
        fi
    done
fi
if [[ -z "${TRTEXEC}" ]]; then
    echo "trtexec is required; install the TensorRT samples/tools package first." >&2
    exit 1
fi
if [[ ! -f "${ENGINE}" ]]; then
    echo "Missing engine: ${ENGINE}; build it in 02_model_convert first." >&2
    exit 1
fi

"${TRTEXEC}" --loadEngine="${ENGINE}" --warmUp=500 --duration=3 --noDataTransfers \
    --profilingVerbosity=detailed --dumpLayerInfo > "${NSYS_DIR}/trtexec-eager.log" 2>&1
nsys profile --trace=cuda,nvtx,osrt --force-overwrite=true \
    --output="${NSYS_DIR}/yolo26n-cls-eager" \
    "${TRTEXEC}" --loadEngine="${ENGINE}" --warmUp=500 --duration=3 --noDataTransfers
nsys profile --trace=cuda,nvtx,osrt --force-overwrite=true \
    --output="${NSYS_DIR}/yolo26n-cls-cudagraph" \
    "${TRTEXEC}" --loadEngine="${ENGINE}" --warmUp=500 --duration=3 --noDataTransfers --useCudaGraph
echo "Nsight Systems reports and trtexec layer log: ${NSYS_DIR}"
