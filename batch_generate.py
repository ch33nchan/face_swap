
import csv
import os
import torch
from diffusers import FluxPipeline
from peft import PeftModel

def main():
    csv_file = "Master - Tech Solutioning - Char Const - Rerun with head_eye gaze.csv"
    output_dir = "batch_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Reading CSV: {csv_file}")
    targets = []
    with open(csv_file, 'r', encoding='utf-8-sig') as f: # utf-8-sig handles BOM if present
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            print("Empty CSV")
            return
            
        # Find 'Fix Char Const' column index
        try:
            col_idx = header.index("Fix Char Const")
        except ValueError:
            print("Column 'Fix Char Const' not found.")
            print(f"Available columns: {header}")
            return

        for idx, row in enumerate(reader):
            if len(row) > col_idx:
                val = row[col_idx].strip().lower()
                if val == 'no':
                    targets.append(idx)

    print(f"Found {len(targets)} rows with 'Fix Char Const' = NO")
    
    if not targets:
        return

    print("Loading FLUX...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Use bfloat16 as per training
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16
    ).to(device)

    print("Loading LoRA...")
    try:
        pipe.transformer = PeftModel.from_pretrained(pipe.transformer, "lora_output/lora_final")
        print("LoRA loaded.")
    except Exception as e:
        print(f"Failed to load LoRA: {e}")
        return
    
    prompt = "photorealistic portrait, high quality, detailed face, natural lighting, professional photography"
    print(f"Using Prompt: {prompt}")
    
    for idx in targets:
        row_num = idx + 2 # Header is row 1
        print(f"Generating for CSV Row {row_num}...")
        
        image = pipe(
            prompt,
            height=1024,
            width=1024,
            num_inference_steps=28,
            guidance_scale=3.5,
        ).images[0]
        
        save_path = os.path.join(output_dir, f"row_{row_num}.png")
        image.save(save_path)
        print(f"Saved {save_path}")

if __name__ == "__main__":
    main()
