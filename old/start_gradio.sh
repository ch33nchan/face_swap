#!/bin/bash
set -e

source .venv/bin/activate

echo "Starting Gradio interface..."
python3 -m src.gradio_app
