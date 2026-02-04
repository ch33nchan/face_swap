import argparse
import os
from pathlib import Path
import torch
from PIL import Image
from src.face_swap import FaceSwapPipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_lora_from_hf(repo_id: str, output_dir: str):
    from huggingface_hub import hf_hub_download
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    hf_token = os.getenv("HF_TOKEN")
    
    logger.info(f"Downloading LORA from {repo_id}")
    
    try:
        lora_path = hf_hub_download(
            repo_id=repo_id,
            filename="pytorch_lora_weights.safetensors",
            local_dir=output_path,
            token=hf_token,
        )
        logger.info(f"Downloaded to {lora_path}")
        return lora_path
    except Exception as e:
        logger.error(f"Download failed: {e}")
        try:
            files = [
                "lora_weights.safetensors",
                "adapter_model.safetensors",
                "diffusion_pytorch_model.safetensors",
            ]
            for filename in files:
                try:
                    lora_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        local_dir=output_path,
                        token=hf_token,
                    )
                    logger.info(f"Downloaded {filename} to {lora_path}")
                    return lora_path
                except:
                    continue
        except Exception as e2:
            logger.error(f"All download attempts failed: {e2}")
            return None


def test_lora(lora_path: str, base_image: str, reference_image: str, output_path: str):
    logger.info(f"Initializing pipeline with LORA: {lora_path}")
    
    pipeline = FaceSwapPipeline(
        model_id="black-forest-labs/FLUX.1-dev",
        device="cuda" if torch.cuda.is_available() else "cpu",
        lora_path=lora_path,
        lora_scale=1.0,
    )
    
    logger.info("Running face swap")
    result = pipeline.swap_face(
        base_image=base_image,
        reference_image=reference_image,
        prompt="high quality portrait, detailed face, natural lighting, photorealistic",
        num_inference_steps=28,
        guidance_scale=3.5,
        strength=0.75,
    )
    
    result.save(output_path)
    logger.info(f"Saved result to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Test pre-trained LORA models")
    parser.add_argument("--repo-id", type=str, help="HuggingFace repo ID (e.g., Alissonerdx/BFS-Best-Face-Swap)")
    parser.add_argument("--lora-path", type=str, help="Local LORA path (alternative to repo-id)")
    parser.add_argument("--base-image", type=str, required=True, help="Base image path")
    parser.add_argument("--reference-image", type=str, required=True, help="Reference image path")
    parser.add_argument("--output", type=str, default="output.png", help="Output image path")
    parser.add_argument("--download-dir", type=str, default="downloaded_loras", help="Directory for downloaded LORAs")
    
    args = parser.parse_args()
    
    if args.repo_id:
        lora_path = download_lora_from_hf(args.repo_id, args.download_dir)
        if not lora_path:
            logger.error("Failed to download LORA")
            return
    elif args.lora_path:
        lora_path = args.lora_path
    else:
        logger.error("Must provide either --repo-id or --lora-path")
        return
    
    test_lora(lora_path, args.base_image, args.reference_image, args.output)


if __name__ == "__main__":
    main()
