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


def test_klein_lora(lora_path: str, base_image: str, reference_image: str, output_path: str):
    logger.info(f"Testing LORA: {lora_path}")
    
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN required for Klein model")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
    from diffusers.models.transformers import FluxTransformer2DModel
    from transformers import AutoTokenizer, T5EncoderModel
    from safetensors.torch import load_file
    
    logger.info("Loading FLUX Klein components...")
    
    # Load VAE
    vae = AutoencoderKL.from_pretrained(
        "black-forest-labs/FLUX.2-klein-4b",
        subfolder="vae",
        torch_dtype=torch.float16,
        token=hf_token,
    ).to(device)
    
    # Load tokenizer with AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "black-forest-labs/FLUX.2-klein-4b",
        subfolder="tokenizer",
        token=hf_token,
    )
    
    # Load text encoder
    text_encoder = T5EncoderModel.from_pretrained(
        "black-forest-labs/FLUX.2-klein-4b",
        subfolder="text_encoder",
        torch_dtype=torch.float16,
        token=hf_token,
    ).to(device)
    
    # Load transformer
    transformer = FluxTransformer2DModel.from_pretrained(
        "black-forest-labs/FLUX.2-klein-4b",
        subfolder="transformer",
        torch_dtype=torch.float16,
        token=hf_token,
    ).to(device)
    
    # Load scheduler
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        "black-forest-labs/FLUX.2-klein-4b",
        subfolder="scheduler",
        token=hf_token,
    )
    
    # Load and inspect LORA weights
    logger.info(f"Loading LORA weights from {lora_path}")
    lora_state_dict = load_file(lora_path)
    
    lora_keys = list(lora_state_dict.keys())
    logger.info(f"LORA has {len(lora_keys)} keys")
    if lora_keys:
        logger.info(f"Sample keys: {lora_keys[:3]}")
    
    # Generate image
    logger.info("Generating image with Klein model")
    
    prompt = "photorealistic portrait, high quality, detailed face, natural lighting"
    
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    
    with torch.no_grad():
        prompt_embeds = text_encoder(text_inputs.input_ids)[0].to(torch.float16)
    
    latent_height = 64
    latent_width = 64
    latents = torch.randn(
        (1, transformer.config.in_channels, latent_height, latent_width),
        device=device,
        dtype=torch.float16,
    )
    
    scheduler.set_timesteps(28)
    
    logger.info("Running denoising loop...")
    for i, t in enumerate(scheduler.timesteps):
        timestep = t.to(device).unsqueeze(0).to(torch.float16)
        
        with torch.no_grad():
            noise_pred = transformer(
                hidden_states=latents,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]
        
        latents = scheduler.step(noise_pred, t, latents).prev_sample
        
        if i % 5 == 0:
            logger.info(f"Step {i+1}/{len(scheduler.timesteps)}")
    
    with torch.no_grad():
        image = vae.decode(latents / vae.config.scaling_factor).sample
    
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
    image = (image * 255).astype("uint8")
    result = Image.fromarray(image)
    
    result.save(output_path)
    logger.info(f"Saved result to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Test BFS LORA with FLUX Klein")
    parser.add_argument("--repo-id", type=str, default="Alissonerdx/BFS-Best-Face-Swap")
    parser.add_argument("--lora-path", type=str)
    parser.add_argument("--base-image", type=str, required=True)
    parser.add_argument("--reference-image", type=str, required=True)
    parser.add_argument("--output", type=str, default="output_klein.png")
    
    args = parser.parse_args()
    
    if args.lora_path:
        lora_path = args.lora_path
    else:
        lora_path = download_klein_lora(args.repo_id)
    
    test_klein_lora(lora_path, args.base_image, args.reference_image, args.output)


if __name__ == "__main__":
    main()
