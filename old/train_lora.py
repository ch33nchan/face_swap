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
    learning_rate: float = 5e-5,
    num_epochs: int = 100,
    batch_size: int = 1,
    save_every: int = 10,
    gpu_id: int = 0,
):
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Use bfloat16 for training to avoid NaNs (FLUX recommended)
    weight_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    logger.info(f"Using dtype: {weight_dtype}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    writer = SummaryWriter(log_dir=output_path / "logs")
    
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logger.warning("HF_TOKEN not set. Some models may require authentication.")
    
    logger.info(f"Loading model: {model_id}")
    pipe = FluxPipeline.from_pretrained(
        model_id,
        torch_dtype=weight_dtype,
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
    
    # Ensure transformer is in fp16
    # Ensure transformer is in correct dtype
    transformer = transformer.to(weight_dtype)
    
    pipe.transformer = transformer
    
    # Ensure VAE is in float32 to avoid NaN/Garbage during encoding
    pipe.vae.to(dtype=torch.float32)
    
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
            # Use float32 for VAE encoding
            images = batch.to(device, dtype=torch.float32)
            
            with torch.no_grad():
                latents = pipe.vae.encode(images).latent_dist.sample()
                latents = (latents * pipe.vae.config.scaling_factor).to(weight_dtype)
            
            # Flow matching: interpolate between noise and latent
            noise = torch.randn_like(latents, dtype=weight_dtype)
            # Timestep for this batch
            timesteps_1d = torch.rand((latents.shape[0],), device=device, dtype=weight_dtype)
            timesteps = timesteps_1d.view(-1, 1, 1, 1)
            
            # Linear interpolation for flow matching
            noisy_latents = (timesteps * latents + (1 - timesteps) * noise).to(weight_dtype)
            
            # Target is the direction from noise to latent
            target = latents - noise
            
            # Get prompt embeddings
            # FLUX uses: CLIP (text_encoder) for pooled_projections, T5 (text_encoder_2) for encoder_hidden_states
            with torch.no_grad():
                # T5 encoder_hidden_states
                t5_output = pipe.text_encoder_2(
                    pipe.tokenizer_2(
                        "A photo of sks person",
                        padding="max_length",
                        max_length=512,
                        truncation=True,
                        return_tensors="pt",
                    ).input_ids.to(device)
                )
                encoder_hidden_states = t5_output[0].to(weight_dtype)
                
                # CLIP pooled embeddings
                clip_output = pipe.text_encoder(
                    pipe.tokenizer(
                        "A photo of sks person",
                        padding="max_length",
                        max_length=pipe.tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    ).input_ids.to(device),
                    output_hidden_states=False,
                )
                pooled_prompt_embeds = clip_output.pooler_output.to(weight_dtype)
            
            # FLUX requires guidance value
            guidance_vec = torch.full((latents.shape[0],), 3.5, device=device, dtype=weight_dtype)
            
            # Timestep for this batch
            # timesteps_1d already created above
            
            # FLUX packing: (B, C, H, W) -> (B, (H//2)*(W//2), C*4)
            # Standard patch_size is 2. Input channels 16 -> 64.
            batch_size = latents.shape[0]
            height, width = latents.shape[2], latents.shape[3]
            patch_size = 2
            channel_dim = noisy_latents.shape[1]
            
            # Pack latents
            hidden_states = noisy_latents.view(batch_size, channel_dim, height // patch_size, patch_size, width // patch_size, patch_size)
            hidden_states = hidden_states.permute(0, 2, 4, 1, 3, 5)
            hidden_states = hidden_states.reshape(batch_size, (height // patch_size) * (width // patch_size), channel_dim * patch_size * patch_size)
            
            # Prepare img_ids for packed latents
            # Grid size is reduced by patch_size
            packed_height = height // patch_size
            packed_width = width // patch_size
            
            img_ids = torch.zeros(packed_height * packed_width, 3, device=device, dtype=weight_dtype)
            img_ids[:, 1] = torch.arange(packed_height, device=device).repeat_interleave(packed_width).to(weight_dtype)
            img_ids[:, 2] = torch.arange(packed_width, device=device).repeat(packed_height).to(weight_dtype)
            
            # Text position IDs (seq_len x 3)
            txt_seq_len = encoder_hidden_states.shape[1]
            txt_ids = torch.zeros(txt_seq_len, 3, device=device, dtype=weight_dtype)
            
            # Pass packed hidden_states
            model_pred = transformer(
                hidden_states=hidden_states,
                timestep=timesteps_1d,
                guidance=guidance_vec,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=encoder_hidden_states,
                txt_ids=txt_ids,
                img_ids=img_ids,
                return_dict=False,
            )[0]
            
            # Unpack model_pred for loss calculation: (B, L, C*4) -> (B, 16, H, W)
            # But wait, loss is computed in latent space.
            # We should pack 'target' (noise) as well to match model_pred shape!
            
            target_packed = target.view(batch_size, channel_dim, height // patch_size, patch_size, width // patch_size, patch_size)
            target_packed = target_packed.permute(0, 2, 4, 1, 3, 5)
            target_packed = target_packed.reshape(batch_size, (height // patch_size) * (width // patch_size), channel_dim * patch_size * patch_size)
            
            # Compute loss
            loss = torch.nn.functional.mse_loss(model_pred, target_packed)
            
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
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU device ID to use")
    
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
        gpu_id=args.gpu_id,
    )


if __name__ == "__main__":
    main()
