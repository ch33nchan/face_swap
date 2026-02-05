#!/bin/bash
if [ -f .env ]; then
    export $(cat .env | xargs)
fi

# Ensure venv is active
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "Starting Character Swap Gradio App..."
python gradio_char_swap.py
