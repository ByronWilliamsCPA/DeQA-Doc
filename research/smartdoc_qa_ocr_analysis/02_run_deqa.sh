#!/bin/bash
# Run DeQA inference on SmartDoc-QA images.
#
# Prerequisites:
#   1. Build meta JSON:  cd DeQA-Score && .venv/bin/python ../research/smartdoc_qa_ocr_analysis/01_build_meta_json.py
#   2. Ensure model weights are available (downloads from HF on first run)
#
# Usage:
#   cd DeQA-Score
#   bash ../research/smartdoc_qa_ocr_analysis/02_run_deqa.sh [GPU_ID]
#
# Uses the Scorer-based iqa_eval.py with --with-prob to get probability distributions.
# Results are appended (resume-safe) — re-run to continue after interruption.

set -euo pipefail

GPU_ID="${1:-0}"
MODEL_PATH="${MODEL_PATH:-zhiyuanyou/DeQA-Score-Mix3}"
BATCH_SIZE="${BATCH_SIZE:-4}"

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEQA_DIR="$(cd "${SCRIPT_DIR}/../../DeQA-Score" && pwd)"
META_DIR="${SCRIPT_DIR}/data"
SAVE_DIR="${SCRIPT_DIR}/data/deqa_results"
ROOT_DIR="/mnt/e/image_detection/02_benchmark_only/smartdoc-qa/Dataset SmartDoc-QA/Captured_Images"

# Verify meta files exist
if [ ! -f "${META_DIR}/smartdoc_qa_meta_all.json" ]; then
    echo "ERROR: Meta JSON not found. Run 01_build_meta_json.py first."
    exit 1
fi

echo "=== SmartDoc-QA DeQA Inference ==="
echo "  GPU:        cuda:${GPU_ID}"
echo "  Model:      ${MODEL_PATH}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Root dir:   ${ROOT_DIR}"
echo "  Save dir:   ${SAVE_DIR}"
echo ""

cd "${DEQA_DIR}"
export PYTHONPATH=./:${PYTHONPATH:-}

mkdir -p "${SAVE_DIR}"

# Run inference on all images
# iqa_eval.py supports resume — it skips images already in the output file.
.venv/bin/python src/evaluate/iqa_eval.py \
    --level-names excellent good fair poor bad \
    --model-path "${MODEL_PATH}" \
    --preprocessor-path ./preprocessor/ \
    --root-dir "${ROOT_DIR}" \
    --meta-paths "${META_DIR}/smartdoc_qa_meta_all.json" \
    --save-dir "${SAVE_DIR}" \
    --device "cuda:${GPU_ID}" \
    --batch-size "${BATCH_SIZE}" \
    --with-prob True

echo ""
echo "=== Inference complete ==="
echo "Results: ${SAVE_DIR}/smartdoc_qa_meta_all.json"
echo ""
echo "Next step: run 03_compute_correlation.py to join DeQA scores with OCR error rates."
