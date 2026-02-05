import torch
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import os
from diffusers import FluxImg2ImgPipeline
from torchvision import transforms

# Configuration
CSV_PATH = "Master - Tech Solutioning - Char Const - Rerun with head_eye gaze.csv"
OUTPUT_DIR = "baseline_outputs"
LORA_PATH = "/mnt/data1/srini/face_swap/downloaded_loras/bfs_head_v1_flux-klein_9b_step3500_rank128.safetensors"
PROMPT = (
    "head_swap: Use image 1 as the base image, preserving its environment, background, camera perspective, framing, exposure, contrast, and lighting. "
    "Remove the head from image 1 and seamlessly replace it with the head from image 2.\n"
    "Match the original head size, face-to-body ratio, neck thickness, shoulder alignment, and camera distance so proportions remain natural and unchanged.\n\n"
    "Adapt the inserted head to the lighting of image 1 by matching light direction, intensity, softness, color temperature, shadows, and highlights, with no independent relighting.\n"
    "Preserve the identity of image 2, including hair texture, eye color, nose structure, facial proportions, and skin details.\n"
    "Match the pose and expression from image 1, including head tilt, rotation, eye direction, gaze, micro-expressions, and lip position.\n"
    "Ensure seamless neck and jaw blending, consistent skin tone, realistic shadow contact, natural skin texture, and uniform sharpness.\n"
    "Photorealistic, high quality, sharp details, 4K."
)

def batch_generate_baseline():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print("Loading FLUX Img2Img (FP32 + CPU Offload)...")
    try:
        pipe = FluxImg2ImgPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.float32
        )
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        return

    print(f"Loading Baseline LoRA from {LORA_PATH}...")
    try:
        # Use load_lora_weights for external LoRA file
        pipe.load_lora_weights(LORA_PATH)
        print("LoRA loaded.")
    except Exception as e:
        print(f"Failed to load LoRA: {e}")
        return
        
    pipe.enable_sequential_cpu_offload()
    
    # Load CSV
    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
        
    print(f"Processing {len(df)} rows...")
    
    for index, row in df.iterrows():
        try:
            print(f"Processing Row {index}...")
            image_url = row['image_url']
            
            # Download Image
            response = requests.get(image_url)
            init_image = Image.open(BytesIO(response.content)).convert("RGB")
            init_image = init_image.resize((1024, 1024)) # Resize for consistency
            
            print("Running generation...")
            # Generate
            image = pipe(
                prompt=PROMPT,
                image=init_image,
                strength=0.75, # Standard Img2Img strength
                guidance_scale=3.5,
                num_inference_steps=28,
                generator=torch.Generator("cpu").manual_seed(42)
            ).images[0]
            
            output_path = os.path.join(OUTPUT_DIR, f"row_{index}.png")
            image.save(output_path)
            print(f"Saved to {output_path}")
            
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            continue

if __name__ == "__main__":
    batch_generate_baseline()
