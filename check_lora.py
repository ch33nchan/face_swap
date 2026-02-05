
import torch
import os
import sys
from safetensors.torch import load_file

def check_lora(path):
    print(f"Checking LoRA at {path}...")
    
    safetensors_path = os.path.join(path, "adapter_model.safetensors")
    bin_path = os.path.join(path, "adapter_model.bin")
    
    if os.path.exists(safetensors_path):
        print(f"Loading {safetensors_path}")
        tensors = load_file(safetensors_path)
    elif os.path.exists(bin_path):
        print(f"Loading {bin_path}")
        tensors = torch.load(bin_path, map_location="cpu")
    else:
        print("No model file found.")
        return

    has_nan = False
    has_inf = False
    max_val = 0.0
    
    for k, v in tensors.items():
        v_float = v.float() # Cast to float32 for checking
        if torch.isnan(v_float).any():
            print(f"NaN found in {k}")
            has_nan = True
        if torch.isinf(v_float).any():
            print(f"Inf found in {k}")
            has_inf = True
            
        avg = v_float.abs().mean().item()
        mx = v_float.abs().max().item()
        if mx > max_val:
            max_val = mx
            
    print(f"Max Weight Value: {max_val}")
    
    if has_nan or has_inf:
        print("LoRA IS CORRUPTED (NaNs/Infs).")
    elif max_val > 10.0:
        print("LoRA weights seem suspiciously large (>10).")
    else:
        print("LoRA weights look completely normal.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_lora.py <lora_path>")
    else:
        check_lora(sys.argv[1])
