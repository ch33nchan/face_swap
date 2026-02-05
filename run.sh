#!/bin/bash

set -e

CSV_PATH="${1:-data/input.csv}"
WORKFLOW_PATH="${2:-workflows/Flux2_Klein_9b_Face_Swap.json}"
OUTPUT_DIR="${3:-./output}"
GPU_ID="${4:-0}"
COMFYUI_URL="${5:-http://localhost:8188}"



python3 comfyui_char_const_pipeline.py \
  --csv "$CSV_PATH" \
  --workflow "$WORKFLOW_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --comfyui-url "$COMFYUI_URL" \
  --gpu-id "$GPU_ID" \
  --fix-column "Fix Char Const" \
  --fix-value "No" \
  --poll-interval 2 \
  --timeout 300

echo "Processing complete. Results in: $OUTPUT_DIR/processing_results.csv"