#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="microzoo-owlv2-v1"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda 未找到，请先初始化 Anaconda/Miniconda。" >&2
    exit 1
fi

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    conda create --yes --override-channels -c conda-forge -n "${ENV_NAME}" \
        python=3.10 numpy=1.26.4 pillow=10.4.0 pip
else
    echo "Conda 环境 ${ENV_NAME} 已存在，保留现有环境。"
fi

ENV_PREFIX="$(conda run -n "${ENV_NAME}" python -c 'import sys; print(sys.prefix)')"
SITE_PACKAGES="${ENV_PREFIX}/lib/python3.10/site-packages"

# Jetson 上的 NVIDIA PyTorch/TensorRT 是系统 Python 3.10 构建。
# 让 Conda Python 3.10 可见系统包，但不在 Conda 中重新安装/覆盖 Torch。
printf '%s\n' \
    '/usr/local/lib/python3.10/dist-packages' \
    '/usr/lib/python3.10/dist-packages' \
    > "${SITE_PACKAGES}/jetson_system_packages.pth"

conda run -n "${ENV_NAME}" python -m pip install -r "${SCRIPT_DIR}/requirements.txt"
conda run -n "${ENV_NAME}" python "${SCRIPT_DIR}/01-check_environment.py"
