#!/usr/bin/env bash
set -euo pipefail

INPUT_ROOT="./raw_data/input_yuv"
COMPRESSED_ROOT="./raw_data/compressed"
OUTPUT_ROOT="./dataset/testset"

QP_VALUES="27,42"
WORKERS=4

for DATASET in CO3Dv2 Free RealEstate10K; do
    echo "Processing ${DATASET}"

    # 1. 모든 scene을 코덱·QP별로 압축하고 복원 YUV 저장
    python compression.py \
        --input "${INPUT_ROOT}/${DATASET}" \
        --output "${COMPRESSED_ROOT}/${DATASET}" \
        --qp-values "${QP_VALUES}" \
        --codec AVC HEVC AV1 VP9 \
        --backend software \
        --workers "${WORKERS}"

    # 2. 압축 작업 성공 여부 확인
    python - "${COMPRESSED_ROOT}/${DATASET}/job_log.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    jobs = json.load(f)

failed = [
    job for job in jobs
    if job.get("status") not in {"ok", "skipped_existing"}
]

if not jobs or failed:
    raise SystemExit(
        f"Compression incomplete: {len(failed)} failed jobs. "
        f"Check {sys.argv[1]}"
    )
PY

    # 3. yuv -> frame
    #    if original RGB frame exist, copy that into GT folder
    #    elif export GT PNG from original yuv.
    python export_frame.py \
        --split test \
        --input "${INPUT_ROOT}/${DATASET}" \
        --compressed "${COMPRESSED_ROOT}/${DATASET}" \
        --output "${OUTPUT_ROOT}" \
        --dataset "${DATASET}" \
        --fps 30
done

echo "Test set preparation completed."