#!/bin/bash

set -e

CSV_PATH="${1:-data/input.csv}"
OUTPUT_DIR="${2:-./output}"
GPU_ID="${3:-0}"
MODEL_ID="${4:-black-forest-labs/FLUX.1-dev}"


python3 comfyui_char_const_pipeline.py \
  --csv "$CSV_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --gpu-id "$GPU_ID" \
  --model-id "$MODEL_ID" \
  --fix-column "Fix Char Const" \
  --fix-value "No" \
  --guidance-scale 3.5 \
  --num-inference-steps 50

echo "Processing complete. Results in: $OUTPUT_DIR/processing_results.csv"