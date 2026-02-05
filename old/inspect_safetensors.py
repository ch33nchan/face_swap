from safetensors.torch import load_file
import sys

path = "/mnt/data1/srini/face_swap/downloaded_loras/bfs_head_v1_flux-klein_9b_step3500_rank128.safetensors"
try:
    tensors = load_file(path)
    print(f"Total keys: {len(tensors)}")
    print("First 20 keys:")
    for k in list(tensors.keys())[:20]:
        print(k)
except Exception as e:
    print(e)
