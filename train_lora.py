import argparse
import os
from pathlib import Path
import torch
from diffusers import FluxPipeline
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor2_0
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import logging
import json
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceDataset(Dataset):
    def __init__(self, image_dir: str, size: int = 512):
        self.image_paths = list(Path(image_dir).glob("*.jpg")) + list(Path(image_dir).glob("*.png"))
        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img)


def train_lora(
    image_dir: str,
    output_dir: str,
    model_id: str = "black-forest-labs/FLUX.1-dev",
    rank: int = 64,
    learning_rate: float = 1e-4,
    num_epochs: int = 100,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    save_every: int = 10,
    device: str = "cuda",
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    writer = SummaryWriter(log_dir=output_path / "logs")
    
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logger.warning("HF_TOKEN not set. Some models may require authentication.")
    
    logger.info(f"Loading model: {model_id}")
    pipe = FluxPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        token=hf_token,
    ).to(device)
    
    unet = pipe.transformer
    unet.requires_grad_(False)
    
    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        lora_attn_procs[name] = LoRAAttnProcessor2_0(
            rank=rank,
            network_alpha=rank,
        )
    
    unet.set_attn_processor(lora_attn_procs)
    
    lora_layers = AttnProcsLayers(unet.attn_processors)
    lora_layers.to(device, dtype=torch.float16)
    
    optimizer = torch.optim.AdamW(lora_layers.parameters(), lr=learning_rate)
    
    dataset = FaceDataset(image_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    logger.info(f"Training on {len(dataset)} images for {num_epochs} epochs")
    
    metrics = {
        "epoch_losses": [],
        "step_losses": [],
        "epochs": [],
        "steps": [],
    }
    
    global_step = 0
    epoch_loss_accum = 0.0
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for step, batch in enumerate(progress_bar):
            batch = batch.to(device, dtype=torch.float16)
            
            latents = pipe.vae.encode(batch).latent_dist.sample()
            latents = latents * pipe.vae.config.scaling_factor
            
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (batch.shape[0],), device=device)
            
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
            
            noise_pred = unet(noisy_latents, timesteps).sample
            
            loss = torch.nn.functional.mse_loss(noise_pred, noise, reduction="mean")
            loss_value = loss.item()
            
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                
                metrics["step_losses"].append(loss_value)
                metrics["steps"].append(global_step)
                writer.add_scalar("Loss/step", loss_value, global_step)
                
                global_step += 1
            
            epoch_loss += loss_value
            progress_bar.set_postfix({"loss": f"{loss_value:.4f}"})
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        metrics["epoch_losses"].append(avg_epoch_loss)
        metrics["epochs"].append(epoch + 1)
        writer.add_scalar("Loss/epoch", avg_epoch_loss, epoch + 1)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_epoch_loss:.4f}")
        
        if (epoch + 1) % save_every == 0 or (epoch + 1) == num_epochs:
            save_path = output_path / f"lora_epoch_{epoch+1}.safetensors"
            pipe.save_lora_weights(save_path, unet=unet)
            logger.info(f"Saved checkpoint to {save_path}")
    
    final_path = output_path / "lora_final.safetensors"
    pipe.save_lora_weights(final_path, unet=unet)
    logger.info(f"Training complete. Final weights saved to {final_path}")
    
    with open(output_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(metrics["epochs"], metrics["epoch_losses"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss per Epoch")
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(metrics["steps"], metrics["step_losses"], alpha=0.6)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss per Step")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path / "training_metrics.png", dpi=150)
    logger.info(f"Saved training plots to {output_path / 'training_metrics.png'}")
    
    writer.close()


def main():
    parser = argparse.ArgumentParser(description="Train LORA for face swap")
    parser.add_argument("--image-dir", type=str, required=True, help="Directory containing training images")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for LORA weights")
    parser.add_argument("--model-id", type=str, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    train_lora(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        model_id=args.model_id,
        rank=args.rank,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_every=args.save_every,
        device=args.device,
    )


if __name__ == "__main__":
    main()
