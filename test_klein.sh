#!/bin/bash
set -e

if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN not set"
    echo "Please run: export HF_TOKEN=\"your_token\""
    exit 1
fi

source .venv/bin/activate

BASE_IMAGE=${1:-"base.jpg"}
REF_IMAGE=${2:-"ref.jpg"}
OUTPUT=${3:-"output_klein.png"}
GPU_ID=${4:-1}

echo "Testing BFS LORA with FLUX Klein..."
echo "Base: $BASE_IMAGE"
echo "Reference: $REF_IMAGE"
echo "GPU ID: $GPU_ID"

# Will download Klein 4b LORA automatically
python3 test_klein_lora.py \
    --base-image "$BASE_IMAGE" \
    --reference-image "$REF_IMAGE" \
    --output "$OUTPUT" \
    --gpu-id "$GPU_ID"

echo "Complete! Output: $OUTPUT"
