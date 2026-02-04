#!/bin/bash
set -e

echo "Setting up Face Swap environment..."

if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

echo "Creating virtual environment..."
uv venv

echo "Installing dependencies..."
source .venv/bin/activate
uv pip install -r requirements.txt

echo "Setup complete. To activate: source .venv/bin/activate"
