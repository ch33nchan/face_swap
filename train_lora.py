import argparse
import os
from pathlib import Path
import logging
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import FluxPipeline
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageDataset(Dataset):
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
    
    transformer = pipe.transformer
    transformer.requires_grad_(False)
    
    logger.info(f"Adding LORA adapters with rank {rank}")
    from peft import LoraConfig,get_peft_model
    
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=0.0,
        bias="none",
    )
    
    transformer = get_peft_model(transformer, lora_config)
    transformer.print_trainable_parameters()
    
    pipe.transformer = transformer
    
    dataset = ImageDataset(image_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=learning_rate)
    
    metrics = {
        "step_losses": [],
        "epoch_losses": [],
        "steps": [],
        "epochs": [],
    }
    
    global_step = 0
    logger.info(f"Starting training for {num_epochs} epochs on {len(dataset)} images")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        transformer.train()
        
        for batch_idx, batch in enumerate(dataloader):
            images = batch.to(device)
            
            with torch.no_grad():
                latents = pipe.vae.encode(images).latent_dist.sample()
                latents = latents * pipe.vae.config.scaling_factor
            
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, pipe.scheduler.config.num_train_timesteps,
                (latents.shape[0],), device=device
            ).long()
            
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
            
            with torch.no_grad():
                prompt_embeds = pipe.text_encoder(
                    torch.zeros((latents.shape[0],), device=device, dtype=torch.long)
                )[0]
            
            model_pred = transformer(
                noisy_latents,
                timesteps,
                prompt_embeds,
                return_dict=False,
            )[0]
            
            loss = torch.nn.functional.mse_loss(model_pred, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_value = loss.item()
            epoch_loss += loss_value
            
            metrics["step_losses"].append(loss_value)
            metrics["steps"].append(global_step)
            writer.add_scalar("Loss/step", loss_value, global_step)
            
            if global_step % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Step {global_step}, Loss: {loss_value:.6f}")
            
            global_step += 1
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        metrics["epoch_losses"].append(avg_epoch_loss)
        metrics["epochs"].append(epoch + 1)
        writer.add_scalar("Loss/epoch", avg_epoch_loss, epoch + 1)
        
        logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.6f}")
        
        if (epoch + 1) % save_every == 0:
            checkpoint_path = output_path / f"lora_epoch_{epoch+1}"
            transformer.save_pretrained(checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    final_path = output_path / "lora_final"
    transformer.save_pretrained(final_path)
    logger.info(f"Training complete. Final LORA saved to {final_path}")
    
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
    parser = argparse.ArgumentParser(description="Train LORA for FLUX face swap")
    parser.add_argument("--image-dir", type=str, required=True, help="Directory containing training images")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for LORA weights")
    parser.add_argument("--model-id", type=str, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
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
        save_every=args.save_every,
        device=args.device,
    )


if __name__ == "__main__":
    main()
