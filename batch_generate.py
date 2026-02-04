
import csv
import os
import torch
import requests
from PIL import Image
from io import BytesIO
from diffusers import FluxImg2ImgPipeline
from peft import PeftModel

def download_image(url):
    try:
        print(f"Downloading {url}...")
        response = requests.get(url, timeout=10)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        # Resize to 1024x1024 for FLUX standard (optional but good for consistency)
        img = img.resize((1024, 1024))
        return img
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return None

def main():
    csv_file = "Master - Tech Solutioning - Char Const - Rerun with head_eye gaze.csv"
    output_dir = "batch_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Reading CSV: {csv_file}")
    targets = []
    
    with open(csv_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            # idx is 0-based index of data rows
            if row.get('Fix Char Const', '').strip().lower() == 'no':
                targets.append({
                    'row_idx': idx + 2, # Match Excel row number (Header=1)
                    'url': row.get('Original Image', '').strip()
                })

    print(f"Found {len(targets)} rows with 'Fix Char Const' = NO")
    
    if not targets:
        return

    print("Loading FLUX Img2Img...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        from diffusers import AutoencoderKL
        # Load VAE in float32 to avoid artifacts
        vae = AutoencoderKL.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            subfolder="vae",
            torch_dtype=torch.float32
        ).to(device)

        pipe = FluxImg2ImgPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            vae=vae,
            torch_dtype=torch.bfloat16
        ).to(device)
        
    except Exception as e:
        print(f"Error loading FluxImg2ImgPipeline: {e}")
        print("Make sure diffusers is updated.")
        return

    print("Loading LoRA...")
    try:
        pipe.transformer = PeftModel.from_pretrained(pipe.transformer, "lora_output/lora_final")
        print("LoRA loaded.")
    except Exception as e:
        print(f"Failed to load LoRA: {e}")
        return
    
    prompt = "photorealistic portrait, high quality, detailed face, natural lighting, professional photography"
    print(f"Using Prompt: {prompt}")
    
    for item in targets:
        row_num = item['row_idx']
        url = item['url']
        
        print(f"Processing Row {row_num}...")
        
        if not url:
            print(f"Skipping Row {row_num}: No URL provided.")
            continue
            
        init_image = download_image(url)
        if init_image is None:
            continue
            
        print("Running generation...")
        try:
            image = pipe(
                prompt=prompt,
                image=init_image,
                strength=0.75, # 0.75 allows fair amount of change while keeping structure
                num_inference_steps=28,
                guidance_scale=3.5,
            ).images[0]
            
            save_path = os.path.join(output_dir, f"row_{row_num}.png")
            image.save(save_path)
            print(f"Saved {save_path}")
        except Exception as e:
            print(f"Generation failed for Row {row_num}: {e}")

if __name__ == "__main__":
    main()
