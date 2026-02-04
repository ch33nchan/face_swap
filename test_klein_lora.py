import argparse
import os
from pathlib import Path
import torch
from PIL import Image
from src.face_swap import FaceSwapPipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_klein_lora(lora_path: str, base_image: str, reference_image: str, output_path: str):
    logger.info(f"Initializing FLUX Klein pipeline with LORA: {lora_path}")
    
    # Use FLUX Klein 9b model for BFS LORA
    pipeline = FaceSwapPipeline(
        model_id="black-forest-labs/FLUX.2-klein-9b",
        device="cuda" if torch.cuda.is_available() else "cpu",
        lora_path=lora_path,
        lora_scale=1.0,
    )
    
    logger.info("Running face swap with Klein model")
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
    parser = argparse.ArgumentParser(description="Test BFS LORA with FLUX Klein")
    parser.add_argument("--lora-path", type=str, required=True, help="Path to BFS LORA file")
    parser.add_argument("--base-image", type=str, required=True, help="Base image path")
    parser.add_argument("--reference-image", type=str, required=True, help="Reference image path")
    parser.add_argument("--output", type=str, default="output_klein.png", help="Output image path")
    
    args = parser.parse_args()
    
    test_klein_lora(args.lora_path, args.base_image, args.reference_image, args.output)


if __name__ == "__main__":
    main()
