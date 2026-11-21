#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METRICS_DIR="${SCRIPT_DIR}/metrics"
mkdir -p "${METRICS_DIR}"

if ! command -v tegrastats >/dev/null 2>&1; then
    echo "tegrastats is required for the Jetson system validation." >&2
    exit 1
fi
if [[ ! -x "${SCRIPT_DIR}/baseline/build/yolo26n_cls_demo" ]]; then
    echo "Build the baseline demo first: cd baseline && cmake -S . -B build && cmake --build build -j2" >&2
    exit 1
fi
if [[ ! -x "${SCRIPT_DIR}/v1/build/yolo26n_cls_v1_demo" ]]; then
    echo "Build the V1 demo first: cd v1 && cmake -S . -B build && cmake --build build -j2" >&2
    exit 1
fi

tegrastats --interval 100 --logfile "${METRICS_DIR}/tegrastats.log" &
TEGRASTATS_PID=$!
trap 'kill "${TEGRASTATS_PID}" 2>/dev/null || true; wait "${TEGRASTATS_PID}" 2>/dev/null || true' EXIT

(
    cd "${SCRIPT_DIR}/baseline"
    ./build/yolo26n_cls_demo
) | tee "${METRICS_DIR}/baseline-output.txt"
(
    cd "${SCRIPT_DIR}/v1"
    ./build/yolo26n_cls_v1_demo
) | tee "${METRICS_DIR}/v1-output.txt"

echo "Demo output and tegrastats log: ${METRICS_DIR}"
