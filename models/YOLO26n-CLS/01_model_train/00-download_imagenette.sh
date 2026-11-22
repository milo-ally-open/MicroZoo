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
    "${SCRIPT_DIR}/../05_model_demo/v1/assets/sample.jpg"
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
    echo "Small validation set already ready: ${SMALL_VAL}"
    exit 0
fi
if [[ ! -f "${ARCHIVE}" ]]; then
    curl --fail --location --retry 3 "${DATA_URL}" --output "${ARCHIVE}"
fi

tar -xzf "${ARCHIVE}" -C "${DATA_DIR}"
python3 "${SCRIPT_DIR}/00-prepare_testset.py"
rm -rf "${DATA_DIR}/imagenette2-160" "${ARCHIVE}"
echo "Small Imagenette validation set ready: ${SMALL_VAL}"
