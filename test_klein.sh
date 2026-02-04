#!/bin/bash
set -e

source .venv/bin/activate

LORA_PATH=${1:-"downloaded_loras/bfs_head_v1_flux-klein_9b_step3500_rank128.safetensors"}
BASE_IMAGE=${2:-"base.jpg"}
REF_IMAGE=${3:-"ref.jpg"}
OUTPUT=${4:-"output_klein.png"}

echo "Testing BFS LORA with FLUX Klein..."
echo "LORA: $LORA_PATH"
echo "Base: $BASE_IMAGE"
echo "Reference: $REF_IMAGE"

python3 test_klein_lora.py \
    --lora-path "$LORA_PATH" \
    --base-image "$BASE_IMAGE" \
    --reference-image "$REF_IMAGE" \
    --output "$OUTPUT"

echo "Complete! Output: $OUTPUT"
