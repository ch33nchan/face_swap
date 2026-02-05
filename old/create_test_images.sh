#!/bin/bash
set -e

echo "Creating test images from training data..."

if [ ! -d "training_data" ]; then
    echo "ERROR: training_data not found"
    exit 1
fi

IMG1=$(find training_data -type f \( -name "*.jpg" -o -name "*.png" \) | head -1)
IMG2=$(find training_data -type f \( -name "*.jpg" -o -name "*.png" \) | tail -1)

if [ -z "$IMG1" ] || [ -z "$IMG2" ]; then
    echo "ERROR: Not enough images in training_data"
    exit 1
fi

cp "$IMG1" base.jpg
cp "$IMG2" ref.jpg

echo "Created test images:"
echo "  base.jpg (from: $IMG1)"
echo "  ref.jpg (from: $IMG2)"
echo ""
echo "Now you can run: ./test_lora.sh Alissonerdx/BFS-Best-Face-Swap base.jpg ref.jpg output.png"
