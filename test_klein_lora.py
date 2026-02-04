import argparse
import os
from pathlib import Path
import torch
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_klein_lora(repo_id: str):
    """Download Klein-compatible LORA from HuggingFace"""
    from huggingface_hub import hf_hub_download, list_repo_files
    
    hf_token = os.getenv("HF_TOKEN")
    output_dir = Path("downloaded_loras")
    output_dir.mkdir(exist_ok=True)
    
    logger.info(f"Listing files in {repo_id}")
    files = list_repo_files(repo_id, token=hf_token)
    
    klein_files = [f for f in files if "klein" in f.lower() and f.endswith(".safetensors")]
    
    if not klein_files:
        raise ValueError(f"No Klein LORA files found in {repo_id}")
    
    target_file = None
    for f in klein_files:
        if "4b" in f:
            target_file = f
            break
    
    if not target_file:
        target_file = klein_files[0]
    
    logger.info(f"Downloading {target_file}")
    lora_path = hf_hub_download(
        repo_id=repo_id,
        filename=target_file,
        local_dir=output_dir,
        token=hf_token,
    )
    
    logger.info(f"Downloaded to {lora_path}")
    return lora_path


def test_klein_lora(lora_path: str, base_image: str, reference_image: str, output_path: str, gpu_id: int = 1):
    """Test Klein LORA using Flux2Pipeline"""
    logger.info(f"Testing LORA: {lora_path}")
    
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN required for Klein model")
    
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    from diffusers import Flux2Pipeline
    
    logger.info("Loading FLUX Klein pipeline...")
    
    # Load with device_map to handle meta tensors
    pipe = Flux2Pipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-4b",
        torch_dtype=torch.float16,
        token=hf_token,
        device_map=device,
    )
    
    # Load LORA weights
    logger.info(f"Loading LORA weights from {lora_path}")
    try:
        pipe.load_lora_weights(lora_path)
        logger.info("LORA weights loaded successfully")
    except Exception as e:
        logger.warning(f"LORA loading warning: {e}")
        logger.info("Continuing without LORA for testing...")
    
    # Generate image
    prompt = "photorealistic portrait, high quality, detailed face, natural lighting, professional photography"
    
    logger.info("Generating image...")
    result = pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        num_inference_steps=28,
        guidance_scale=3.5,
    ).images[0]
    
    result.save(output_path)
    logger.info(f"Saved result to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Test BFS LORA with FLUX Klein")
    parser.add_argument("--repo-id", type=str, default="Alissonerdx/BFS-Best-Face-Swap")
    parser.add_argument("--lora-path", type=str)
    parser.add_argument("--base-image", type=str, required=True)
    parser.add_argument("--reference-image", type=str, required=True)
    parser.add_argument("--output", type=str, default="output_klein.png")
    parser.add_argument("--gpu-id", type=int, default=1, help="GPU device ID to use")
    
    args = parser.parse_args()
    
    if args.lora_path:
        lora_path = args.lora_path
    else:
        lora_path = download_klein_lora(args.repo_id)
    
    test_klein_lora(lora_path, args.base_image, args.reference_image, args.output, args.gpu_id)


if __name__ == "__main__":
    main()
