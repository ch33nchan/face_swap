#!/bin/bash
set -e

source .venv/bin/activate

IMAGE_DIR=${1:-"training_data"}
OUTPUT_DIR=${2:-"lora_output"}
EPOCHS=${3:-100}
RANK=${4:-64}

echo "Starting LORA training..."
echo "Image directory: $IMAGE_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Epochs: $EPOCHS"
echo "Rank: $RANK"

python3 train_lora.py \
    --image-dir "$IMAGE_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --rank "$RANK" \
    --lr 1e-4 \
    --batch-size 1 \
    --gradient-accumulation-steps 4 \
    --save-every 10 \
    --device cuda

echo "Training complete. Check $OUTPUT_DIR for results and metrics."
