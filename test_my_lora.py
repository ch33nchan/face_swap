
import os
import torch
import argparse
from diffusers import FluxPipeline
from huggingface_hub import login

def test_my_lora(lora_path, prompt, output_path, gpu_id):
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    print("Loading FLUX.1-dev...")
    # Load standard FLUX pipeline
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
        token=hf_token
    ).to(device)

    print(f"Loading LoRA from {lora_path}...")
    try:
        pipe.load_lora_weights(lora_path)
    except Exception as e:
        print(f"Error loading LoRA: {e}")
        return

    print("Generating image...")
    image = pipe(
        prompt,
        height=1024,
        width=1024,
        num_inference_steps=28,
        guidance_scale=3.5,
    ).images[0]

    image.save(output_path)
    print(f"Saved result to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test trained LoRA with FLUX.1-dev")
    parser.add_argument("lora_path", type=str, help="Path to trained LoRA folder (e.g. lora_output/lora_final)")
    parser.add_argument("prompt", type=str, help="Prompt to generate")
    parser.add_argument("output", type=str, help="Output filename")
    parser.add_argument("gpu_id", type=int, default=0, help="GPU ID to use")
    
    args = parser.parse_args()
    test_my_lora(args.lora_path, args.prompt, args.output, args.gpu_id)
