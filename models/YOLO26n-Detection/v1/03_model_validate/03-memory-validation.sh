#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METRICS_DIR="${SCRIPT_DIR}/metrics/memory"
mkdir -p "${METRICS_DIR}"

if ! command -v tegrastats >/dev/null 2>&1; then
    echo "tegrastats 未安装；请在 Jetson 上运行此脚本。" >&2
    exit 1
fi

tegrastats --interval 1000 --logfile "${METRICS_DIR}/tegrastats.log" &
TEGRASTATS_PID=$!
trap 'kill -TERM "${TEGRASTATS_PID}" 2>/dev/null || true; wait "${TEGRASTATS_PID}" 2>/dev/null || true' EXIT
python3 "${SCRIPT_DIR}/02-task-validate.py"
echo "tegrastats 内存日志：${METRICS_DIR}/tegrastats.log"
