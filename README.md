# Face Swap with FLUX + LORA

Production face swap system with character transfer and LORA fine-tuning.

## Quick Start

### 1. Download Reference Images

Create `links.txt` with image URLs (one per line), then:

```bash
source .venv/bin/activate
python3 download_references.py --links links.txt --output training_data
```

### 2. GPU Setup (One Command)

```bash
git clone <your-repo>
cd char_swap
./setup_gpu.sh
```

### 3. Train LORA

```bash
./start_training.sh training_data lora_output 100 64
```

### 4. Test with Gradio

```bash
./start_gradio.sh
```

Access at `http://localhost:7860`

## Dataset Requirements

**Quantity**: 20-50 images of the character face you want to swap into scenes

**Quality**:
- 512x512+ resolution
- Clear, well-lit faces
- Multiple angles (frontal, profile, 3/4)
- Different expressions (neutral, happy, concerned, angry)
- Minimal occlusions

## Training Outputs

After training in `lora_output/`:
- `lora_final.safetensors` - Final weights
- `training_metrics.png` - Loss plots
- `metrics.json` - Raw metrics
- `logs/` - Tensorboard logs

View tensorboard: `tensorboard --logdir lora_output/logs`

## API

Start server:
```bash
source .venv/bin/activate
python3 run_api.py --host 0.0.0.0 --port 8000
```

Endpoints:
- `POST /swap` - Face swap
- `POST /init` - Load LORA
- `GET /health` - Status

## CLI

```bash
source .venv/bin/activate
python3 src/cli.py \
  --base-image base.jpg \
  --reference-image ref.jpg \
  --output result.jpg \
  --lora-path lora_output/lora_final.safetensors
```

## Structure

```
src/
├── face_swap.py      # Core pipeline
├── api.py            # FastAPI server
├── cli.py            # CLI
└── gradio_app.py     # UI
train_lora.py         # LORA training
download_references.py # Download training images
setup_gpu.sh          # Setup
start_training.sh     # Train
start_gradio.sh       # UI
```
