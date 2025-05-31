import argparse
import os
import shutil
import yaml
from pathlib import Path
from PIL import Image # Ensure PIL is imported

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from diffusers import UNet2DModel, DDPMScheduler
from diffusers.optimization import get_scheduler as get_lr_scheduler
from diffusers.utils import make_image_grid

from tqdm.auto import tqdm
import wandb

# Project-specific imports
from mue.data_handling.datasets import get_gmm_dataloader
from mue.utils.seeding import set_global_seeds
from mue.utils.logging import initialize_wandb # Assuming this utility exists

def load_config(config_path: str) -> dict:
    """Loads and resolves the training configuration from a YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    # Resolve sample_size from rasterization_config if using placeholder
    if isinstance(config['unet_config']['sample_size'], str) and \
       config['unet_config']['sample_size'] == "${rasterization_config.image_size_h_w}":
        config['unet_config']['sample_size'] = tuple(config['rasterization_config']['image_size_h_w'])
    return config

def save_model_checkpoint(model, scheduler, output_dir, epoch_or_step, is_final=False):
    """Saves the model and scheduler state to a checkpoint directory."""
    save_path = Path(output_dir) / f"checkpoint_{'final' if is_final else epoch_or_step}"
    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_path))
    scheduler.save_config(str(save_path / "scheduler_config.json")) # Schedulers save config differently
    print(f"Saved model checkpoint to {save_path}")

def generate_and_save_samples(model, scheduler, num_samples, image_size, num_inference_steps, device, output_path, epoch_or_step):
    """Generates samples from the trained model and saves them as an image grid."""
    model.eval()
    # Ensure image_size is a tuple of integers if it comes from config as list
    if isinstance(image_size, list):
        image_size = tuple(image_size)
        
    # Input shape for UNet2DModel: (batch_size, num_channels, height, width)
    # Our density map is single channel (num_channels=1)
    latents = torch.randn(
        (num_samples, model.config.in_channels, image_size[0], image_size[1]),
        device=device,
        dtype=model.dtype # Use model's dtype
    )

    scheduler.set_timesteps(num_inference_steps)

    for t in tqdm(scheduler.timesteps, desc="Sampling for evaluation"):
        with torch.no_grad():
            model_output = model(latents, t).sample
        latents = scheduler.step(model_output, t, latents).prev_sample

    # Post-process samples: latents are now "clean" images (density maps)
    # Assuming density maps were normalized to [-1, 1]
    samples = (latents / 2 + 0.5).clamp(0, 1)  # Denormalize to [0, 1]
    samples = (samples * 255).type(torch.uint8).cpu() # Convert to 0-255 for image saving

    # --- FIX: Convert PyTorch tensors to PIL Images ---
    pil_images = []
    for img_tensor in samples:
        # Squeeze the channel dimension (from 1, H, W to H, W) and convert to NumPy array
        # Then create a PIL Image in 'L' (grayscale) mode.
        pil_images.append(Image.fromarray(img_tensor.squeeze(0).numpy(), mode='L'))
    
    grid = make_image_grid(pil_images, rows=1, cols=num_samples) # Pass the list of PIL Images
    
    # Ensure output_path's parent directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)
    print(f"Saved sample grid to {output_path}")
    return grid # Return PIL image for W&B logging

def main(config_path: str):
    """Main training function for the DDPM."""
    config = load_config(config_path)

    # Setup output directory
    output_dir = Path(config['training_config']['output_dir'])
    if output_dir.exists() and config['training_config'].get('overwrite_output_dir', False):
        print(f"Overwriting output directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the config file to the output directory for reproducibility
    with open(output_dir / "training_config.yaml", 'w') as f:
        yaml.dump(config, f)

    # --- Initialize ---
    set_global_seeds(config['training_config']['seed'])
    device = torch.device(config['training_config']['device'] if torch.cuda.is_available() else "cpu")
    
    # Determinism flags (as discussed in Report I.D)
    if device.type == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True) # Can be too restrictive, enable if needed and test

    # Initialize W&B
    run_name = f"{config['wandb_config']['run_name_prefix']}_seed{config['training_config']['seed']}"
    wandb_run = initialize_wandb(
        project_name=config['wandb_config']['project_name'],
        run_name=run_name,
        config=config, # Log the entire config
        entity=config['wandb_config'].get('entity')
    )

    # --- Load Data ---
    # Update GMM config with the global seed for data generation consistency
    config['gmm_data_config']['random_state'] = config['training_config']['seed']
    
    train_dataloader, gmm_info = get_gmm_dataloader(
        gmm_config=config['gmm_data_config'],
        image_size=tuple(config['rasterization_config']['image_size_h_w']),
        batch_size=config['dataloader_config']['batch_size'],
        shuffle=config['dataloader_config']['shuffle'],
        num_workers=config['dataloader_config']['num_workers'],
        pin_memory=config['dataloader_config']['pin_memory'],
        use_all_points_for_map=False # Train on the density map with the missing mode
    )
    print(f"Training with GMM config: {gmm_info['config']}")
    print(f"Training data range: {gmm_info['data_range_xy']}")

    # --- Initialize Model, Scheduler, Optimizer ---
    model = UNet2DModel(**config['unet_config']).to(device)
    noise_scheduler = DDPMScheduler(**config['scheduler_config'])
    optimizer = AdamW(
        model.parameters(),
        lr=config['training_config']['learning_rate'],
        betas=(config['training_config']['adam_beta1'], config['training_config']['adam_beta2']),
        weight_decay=config['training_config']['adam_weight_decay'],
        eps=config['training_config']['adam_epsilon'],
    )

    # Calculate total training steps
    # Since the dataset has 1 item (the density map), len(train_dataloader) will be 1.
    # The effective number of times we process this single image is num_train_epochs * (dataloader_batch_size if repeated).
    # Here, our DataLoader will yield one batch (containing the single image).
    num_update_steps_per_epoch = len(train_dataloader) // config['training_config']['gradient_accumulation_steps']
    if num_update_steps_per_epoch == 0: num_update_steps_per_epoch = 1

    if 'num_train_steps' in config['training_config']:
        max_train_steps = config['training_config']['num_train_steps']
        num_train_epochs = max_train_steps // num_update_steps_per_epoch
        if max_train_steps % num_update_steps_per_epoch != 0:
             num_train_epochs +=1 # ensure all steps are covered
    else:
        num_train_epochs = config['training_config']['num_train_epochs']
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
        
    lr_scheduler = get_lr_scheduler(
        name=config['training_config']['lr_scheduler_type'],
        optimizer=optimizer,
        num_warmup_steps=config['training_config']['lr_warmup_steps'] * config['training_config']['gradient_accumulation_steps'],
        num_training_steps=max_train_steps * config['training_config']['gradient_accumulation_steps'],
    )

    if config['training_config']['use_amp']:
        scaler = torch.cuda.amp.GradScaler()

    # --- Training Loop ---
    print("***** Starting DDPM Training on GMM Density Map *****")
    print(f"  Num epochs = {num_train_epochs}")
    print(f"  Num optimization steps = {max_train_steps}")
    print(f"  Instantaneous batch size per device = {config['dataloader_config']['batch_size']}")
    print(f"  Gradient Accumulation steps = {config['training_config']['gradient_accumulation_steps']}")
    print(f"  Effective optimization batch size = {config['dataloader_config']['batch_size'] * config['training_config']['gradient_accumulation_steps']}")

    global_step = 0
    for epoch in range(num_train_epochs):
        model.train()
        progress_bar = tqdm(total=num_update_steps_per_epoch, desc=f"Epoch {epoch+1}/{num_train_epochs}", leave=False)
        epoch_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            clean_images = batch.to(device)
            
            # If dataloader_config.batch_size > 1 was intended for multiple *copies* of the image:
            if clean_images.shape[0] == 1 and config['dataloader_config']['batch_size'] > 1:
                clean_images = clean_images.repeat(config['dataloader_config']['batch_size'], 1, 1, 1)

            noise = torch.randn_like(clean_images)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (clean_images.shape[0],), device=device
            ).long()

            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            if config['training_config']['use_amp']:
                with torch.cuda.amp.autocast():
                    model_output = model(noisy_images, timesteps).sample
                    loss = F.mse_loss(model_output, noise)
                
                scaler.scale(loss / config['training_config']['gradient_accumulation_steps']).backward()
            else:
                model_output = model(noisy_images, timesteps).sample
                loss = F.mse_loss(model_output, noise)
                (loss / config['training_config']['gradient_accumulation_steps']).backward()

            epoch_loss += loss.item() / config['training_config']['gradient_accumulation_steps']

            if (step + 1) % config['training_config']['gradient_accumulation_steps'] == 0:
                if config['training_config']['use_amp']:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                lr_scheduler.step()
                optimizer.zero_grad()
                
                progress_bar.update(1)
                global_step += 1
                
                wandb_run.log({"train_loss_step": loss.item(), "lr": lr_scheduler.get_last_lr()[0], "global_step": global_step})

        avg_epoch_loss = epoch_loss / (step + 1)
        wandb_run.log({"train_loss_epoch": avg_epoch_loss, "epoch": epoch + 1})
        progress_bar.set_postfix({"loss": f"{avg_epoch_loss:.4f}"})
        progress_bar.close()

        # --- Evaluation and Saving ---
        if (epoch + 1) % config['training_config']['save_images_epochs'] == 0 or (epoch + 1) == num_train_epochs:
            sample_output_path = output_dir / "samples" / f"epoch_{epoch+1}_samples.png"
            pil_image_grid = generate_and_save_samples(
                model, noise_scheduler,
                config['training_config']['num_images_per_eval_save'],
                tuple(config['rasterization_config']['image_size_h_w']), # Ensure tuple
                config['training_config']['num_inference_steps_eval'],
                device, sample_output_path, epoch + 1
            )
            if config['wandb_config']['log_images_to_wandb']:
                wandb_run.log({"generated_samples": wandb.Image(pil_image_grid), "epoch": epoch + 1, "global_step": global_step})

        if (epoch + 1) % config['training_config']['save_model_epochs'] == 0 or (epoch + 1) == num_train_epochs:
            save_model_checkpoint(model, noise_scheduler, output_dir, epoch_or_step=epoch + 1)

    # Save final model
    save_model_checkpoint(model, noise_scheduler, output_dir, epoch_or_step="final", is_final=True)
    print("Training finished.")
    wandb_run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DDPM on GMM Density Map.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()
    main(args.config)