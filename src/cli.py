import argparse
import os
from pathlib import Path
from PIL import Image
import logging

from src.face_swap import FaceSwapPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Face swap using FLUX and InsightFace")
    parser.add_argument("--base-image", type=str, required=True, help="Path to base image")
    parser.add_argument("--reference-image", type=str, required=True, help="Path to reference face image")
    parser.add_argument("--output", type=str, required=True, help="Output path for swapped image")
    parser.add_argument("--prompt", type=str, default="high quality portrait, detailed face, natural lighting")
    parser.add_argument("--negative-prompt", type=str, default="blurry, low quality, distorted face, artifacts")
    parser.add_argument("--model-id", type=str, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--lora-path", type=str, default=None, help="Path to LORA weights")
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=28, help="Number of inference steps")
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--strength", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    
    args = parser.parse_args()
    
    if not os.path.exists(args.base_image):
        raise FileNotFoundError(f"Base image not found: {args.base_image}")
    if not os.path.exists(args.reference_image):
        raise FileNotFoundError(f"Reference image not found: {args.reference_image}")
    
    logger.info("Initializing pipeline")
    pipeline = FaceSwapPipeline(
        model_id=args.model_id,
        device=args.device,
        lora_path=args.lora_path,
        lora_scale=args.lora_scale,
    )
    
    logger.info("Performing face swap")
    result = pipeline.swap_face(
        base_image=args.base_image,
        reference_image=args.reference_image,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        strength=args.strength,
        seed=args.seed,
    )
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)
    logger.info(f"Saved result to {output_path}")
    
    pipeline.unload()


if __name__ == "__main__":
    main()
