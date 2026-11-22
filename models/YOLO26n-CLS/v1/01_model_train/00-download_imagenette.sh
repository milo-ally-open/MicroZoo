#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/datasets"
ARCHIVE="${DATA_DIR}/imagenette2-160.tgz"
DATA_URL="https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
SMALL_VAL="${DATA_DIR}/imagenette2-small/val"
SELECTION_FILE="${DATA_DIR}/imagenette2-small/selection.txt"
SAMPLE_COPIES=(
    "${SCRIPT_DIR}/sample.jpg"
    "${SCRIPT_DIR}/../03_model_validate/sample.jpg"
    "${SCRIPT_DIR}/../05_model_demo/assets/sample.jpg"
)

mkdir -p "${DATA_DIR}"
SAMPLES_READY=true
for sample_copy in "${SAMPLE_COPIES[@]}"; do
    if [[ ! -f "${sample_copy}" ]]; then
        SAMPLES_READY=false
    fi
done
if [[ -d "${SMALL_VAL}" ]] && [[ -f "${SELECTION_FILE}" ]] && \
    [[ "$(find "${SMALL_VAL}" -type f | wc -l)" -eq 100 ]] && [[ "${SAMPLES_READY}" == true ]]; then
    echo "ImageNette 固定验证集已就绪：${SMALL_VAL}"
    echo "图片数：$(find "${SMALL_VAL}" -type f | wc -l)，类别数：$(find "${SMALL_VAL}" -mindepth 1 -maxdepth 1 -type d | wc -l)"
    echo "样本清单：${SELECTION_FILE}"
    for sample_copy in "${SAMPLE_COPIES[@]}"; do
        echo "固定样例：${sample_copy}"
    done
    exit 0
fi
if [[ ! -f "${ARCHIVE}" ]]; then
    curl --fail --location --retry 3 "${DATA_URL}" --output "${ARCHIVE}"
fi

tar -xzf "${ARCHIVE}" -C "${DATA_DIR}"
python3 "${SCRIPT_DIR}/00-prepare_testset.py"
rm -rf "${DATA_DIR}/imagenette2-160" "${ARCHIVE}"
echo "ImageNette 固定验证集准备完成：${SMALL_VAL}"
echo "图片数：$(find "${SMALL_VAL}" -type f | wc -l)，类别数：$(find "${SMALL_VAL}" -mindepth 1 -maxdepth 1 -type d | wc -l)"
echo "样本清单：${SELECTION_FILE}"
