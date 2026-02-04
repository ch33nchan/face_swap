#!/bin/bash
set -e

source .venv/bin/activate

REPO_ID=${1:-"Alissonerdx/BFS-Best-Face-Swap"}
BASE_IMAGE=${2:-"base.jpg"}
REF_IMAGE=${3:-"reference.jpg"}
OUTPUT=${4:-"output_pretrained.png"}

echo "Testing pre-trained LORA: $REPO_ID"
echo "Base image: $BASE_IMAGE"
echo "Reference image: $REF_IMAGE"

python3 test_pretrained_lora.py \
    --repo-id "$REPO_ID" \
    --base-image "$BASE_IMAGE" \
    --reference-image "$REF_IMAGE" \
    --output "$OUTPUT"

echo "Result saved to: $OUTPUT"
