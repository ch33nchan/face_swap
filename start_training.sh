#!/bin/bash
set -e

source .venv/bin/activate

IMAGE_DIR=${1:-"training_data"}
OUTPUT_DIR=${2:-"lora_output"}
EPOCHS=${3:-100}
RANK=${4:-64}
GPU_ID=${5:-0}

echo "Starting LORA training..."
echo "Image directory: $IMAGE_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Epochs: $EPOCHS"
echo "Rank: $RANK"
echo "GPU ID: $GPU_ID"

python3 train_lora.py \
    --image-dir "$IMAGE_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --rank "$RANK" \
    --gpu-id "$GPU_ID" \
    --lr 1e-4 \
    --batch-size 1 \
    --save-every 10 \
    --device cuda

echo "Training complete. Check $OUTPUT_DIR for results and metrics."
