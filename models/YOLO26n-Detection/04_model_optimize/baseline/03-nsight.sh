#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE="${SCRIPT_DIR}/engine/yolo26n_fp16.engine"
PROFILE_DIR="${SCRIPT_DIR}/profiling"
NSYS_DIR="${PROFILE_DIR}/nsys"
mkdir -p "${NSYS_DIR}"

if ! command -v nsys >/dev/null 2>&1; then
    echo "nsys 未安装；请先在 Jetson 安装 Nsight Systems。" >&2
    exit 1
fi
TRTEXEC="$(command -v trtexec || true)"
if [[ -z "${TRTEXEC}" ]]; then
    for candidate in /usr/src/tensorrt/bin/trtexec /usr/bin/trtexec; do
        if [[ -x "${candidate}" ]]; then TRTEXEC="${candidate}"; break; fi
    done
fi
if [[ -z "${TRTEXEC}" ]]; then
    echo "trtexec 未安装；请安装 TensorRT samples/tools。" >&2
    exit 1
fi
if [[ ! -f "${ENGINE}" ]]; then
    echo "缺少 engine：${ENGINE}；先构建 02_model_convert engine。" >&2
    exit 1
fi

"${TRTEXEC}" --loadEngine="${ENGINE}" --warmUp=500 --duration=3 --noDataTransfers \
    --profilingVerbosity=detailed --dumpLayerInfo > "${NSYS_DIR}/trtexec-eager.log" 2>&1
nsys profile --trace=cuda,nvtx,osrt --force-overwrite=true \
    --output="${NSYS_DIR}/yolo26n-eager" \
    "${TRTEXEC}" --loadEngine="${ENGINE}" --warmUp=500 --duration=3 --noDataTransfers
nsys profile --trace=cuda,nvtx,osrt --force-overwrite=true \
    --output="${NSYS_DIR}/yolo26n-cudagraph" \
    "${TRTEXEC}" --loadEngine="${ENGINE}" --warmUp=500 --duration=3 --noDataTransfers --useCudaGraph
echo "Nsight Systems 报告和 TensorRT layer 日志：${NSYS_DIR}"
