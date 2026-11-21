#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEMORY_DIR="${SCRIPT_DIR}/metrics/memory"
mkdir -p "${MEMORY_DIR}"

if command -v tegrastats >/dev/null 2>&1; then
    tegrastats --interval 100 --logfile "${MEMORY_DIR}/tegrastats.log" &
    TEGRASTATS_PID=$!
    trap 'kill "${TEGRASTATS_PID}" 2>/dev/null || true; wait "${TEGRASTATS_PID}" 2>/dev/null || true' EXIT
    python3 "${SCRIPT_DIR}/01-numerical-validate.py"
    python3 "${SCRIPT_DIR}/02-task-validate.py"
else
    echo "tegrastats unavailable; running validations without system RAM sampling." >&2
    python3 "${SCRIPT_DIR}/01-numerical-validate.py"
    python3 "${SCRIPT_DIR}/02-task-validate.py"
fi
echo "Validation memory log: ${MEMORY_DIR}/tegrastats.log"
