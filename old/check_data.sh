#!/bin/bash
set -e

echo "Checking training data..."

if [ ! -d "training_data" ]; then
    echo "ERROR: training_data directory not found"
    echo "Please create it and add your reference character images"
    exit 1
fi

IMAGE_COUNT=$(find training_data -type f \( -name "*.jpg" -o -name "*.png" \) | wc -l)

if [ "$IMAGE_COUNT" -eq 0 ]; then
    echo "ERROR: No images found in training_data/"
    echo ""
    echo "Please add 20-50 images of your reference character to training_data/"
    echo "Supported formats: .jpg, .png"
    echo ""
    echo "Directory contents:"
    ls -lah training_data/
    exit 1
fi

echo "Found $IMAGE_COUNT images in training_data/"
echo ""
echo "Sample files:"
find training_data -type f \( -name "*.jpg" -o -name "*.png" \) | head -5

echo ""
echo "Ready to train!"
