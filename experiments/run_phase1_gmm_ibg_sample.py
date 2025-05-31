import argparse
import os
import shutil
import yaml
from pathlib import Path
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import wandb
from typing import Dict, Any, Tuple, Optional, List # <--- ADDED THIS LINE

# Project-specific imports
from mue.data_handling.synthetic_generators import get_gmm_data_for_training_and_evaluation
from mue.data_handling.rasterizers import rasterize_points_to_density_map
from mue.pipelines.ibg_sampling_loop import ibg_sample # Import our new IBG sampler
from mue.utils.seeding import set_global_seeds
from mue.utils.logging import initialize_wandb

# Diffusers imports for model and scheduler
from diffusers import UNet2DModel, DDPMScheduler
from diffusers.utils import make_image_grid # For saving generated samples

def load_config(config_path: str) -> dict:
    """Loads a YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_trained_model_and_scheduler(model_load_config: dict, device: torch.device):
    """Loads a trained UNet2DModel and DDPMScheduler."""
    model_path = Path(model_load_config['trained_model_output_dir']) / "checkpoint_final"
    if not model_path.exists():
        print(f"Warning: 'checkpoint_final' not found at {model_path}. Trying to load directly from {model_load_config['trained_model_output_dir']}.")
        model_path = Path(model_load_config['trained_model_output_dir'])

    if not model_path.exists():
        raise FileNotFoundError(f"Trained model checkpoint not found at {model_path}. Please check 'trained_model_output_dir' in config.")

    model = UNet2DModel.from_pretrained(str(model_path)).to(device)
    print(f"Loaded UNet model from: {model_path}")

    scheduler_config_path = model_path / "scheduler_config.json"
    if not scheduler_config_path.exists():
        print(f"Warning: scheduler_config.json not found at {scheduler_config_path}. Initializing DDPMScheduler with default parameters.")
        scheduler = DDPMScheduler()
    else:
        scheduler = DDPMScheduler.from_config(str(scheduler_config_path))
    print(f"Loaded DDPMScheduler config from: {scheduler_config_path if scheduler_config_path.exists() else 'defaults'}")

    return model, scheduler

def plot_generated_gmm_sample(
    generated_map: np.ndarray,
    gmm_data_info: Dict[str, Any],
    sample_idx: int,
    image_size: Tuple[int, int],
    save_path: Optional[Path] = None,
    dpi: int = 100
):
    """
    Plots a single generated density map, overlaid with GMM mode centers and points.
    Highlights the missing mode.
    """
    fig, ax = plt.subplots(figsize=(8, 7), dpi=dpi)

    (min_x, max_x), (min_y, max_y) = gmm_data_info['data_range_xy']
    extent = [min_x, max_x, min_y, max_y]

    # Plot the generated density map
    # Assuming generated_map is already normalized [0, 1] for visualization
    im = ax.imshow(generated_map, cmap='viridis', origin='lower', extent=extent)
    cbar = fig.colorbar(im, ax=ax, label='Generated Density Value', shrink=0.7)

    # Overlay GMM points and centers for context
    mode_centers_x = [c[0] for c in gmm_data_info['mode_centers']]
    mode_centers_y = [c[1] for c in gmm_data_info['mode_centers']]
    ax.scatter(mode_centers_x, mode_centers_y, marker='X', color='white', s=200, edgecolor='black', linewidth=1.5, zorder=3, label='All Mode Centers')

    if gmm_data_info['missing_mode_center'] is not None:
        # Plot training points (excluding the missing mode)
        ax.scatter(gmm_data_info['training_points'][:, 0], gmm_data_info['training_points'][:, 1],
                   c=gmm_data_info['training_labels'], cmap='viridis', s=5, alpha=0.3, zorder=1, label='Training Points')

        # Plot missing mode points
        if len(gmm_data_info['missing_mode_points']) > 0:
            ax.scatter(gmm_data_info['missing_mode_points'][:, 0], gmm_data_info['missing_mode_points'][:, 1],
                       color='red', s=10, alpha=0.6, zorder=2, label=f"Missing Mode {gmm_data_info['config']['missing_mode_idx']} Points")
            # Highlight missing mode center with a larger marker
            missing_center = gmm_data_info['missing_mode_center']
            ax.scatter(missing_center[0], missing_center[1], marker='*', color='gold', s=400, edgecolor='black', linewidth=2, zorder=4, label=f"Missing Mode {gmm_data_info['config']['missing_mode_idx']} Center")
    else:
        ax.scatter(gmm_data_info['all_points'][:, 0], gmm_data_info['all_points'][:, 1],
                   c=gmm_data_info['all_labels'], cmap='viridis', s=5, alpha=0.3, zorder=1, label='All GMM Points')


    ax.set_title(f"Generated Sample {sample_idx} with IBG (Overlay with GMM)")
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='upper right', fontsize='small')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()

def main(config_path: str):
    config = load_config(config_path)

    # Setup output directory
    output_dir = Path(config['evaluation_config']['output_dir'])
    if output_dir.exists():
        print(f"Warning: Output directory {output_dir} already exists. Contents might be overwritten.")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the config file to the output directory for reproducibility
    with open(output_dir / "ibg_config.yaml", 'w') as f:
        yaml.dump(config, f)

    # --- Initialize ---
    set_global_seeds(config['sampling_config']['seed'])
    device = torch.device(config['evaluation_config']['device'] if torch.cuda.is_available() else "cpu")

    if device.type == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Initialize W&B
    run_name = f"{config['wandb_config']['run_name_prefix']}_scale{config['ibg_config']['guidance_scale']}_seed{config['sampling_config']['seed']}"
    wandb_run = initialize_wandb(
        project_name=config['wandb_config']['project_name'],
        run_name=run_name,
        config=config,
        entity=config['wandb_config'].get('entity')
    )

    # --- Load Trained Model and Scheduler ---
    model, scheduler = load_trained_model_and_scheduler(config['model_load_config'], device)

    # --- Generate GMM Data for Reference (ensure consistency with training) ---
    gmm_data_config_for_ref = config['gmm_data_config'].copy()
    gmm_data_config_for_ref['random_state'] = config['sampling_config']['seed']

    # IMPORTANT: Remove 'missing_mode_idx' from the dictionary
    # so it doesn't conflict with the explicit 'missing_mode_idx=None' argument below.
    if 'missing_mode_idx' in gmm_data_config_for_ref:
        del gmm_data_config_for_ref['missing_mode_idx'] # <--- MODIFIED LINE

    # Generate full GMM data for reference and plotting
    gmm_info_full = get_gmm_data_for_training_and_evaluation(
        **gmm_data_config_for_ref,
        missing_mode_idx=None # Now this will be the ONLY 'missing_mode_idx' passed
    )
    print(f"Reference GMM data range: {gmm_info_full['data_range_xy']}")

    # Log initial maps to W&B if enabled
    if config['evaluation_config']['log_gmm_info_to_wandb']:
        # Also rasterize the training density map for logging
        gmm_info_training = get_gmm_data_for_training_and_evaluation(
            **gmm_data_config_for_ref, # Use gmm_data_config_for_ref (which now lacks missing_mode_idx)
            missing_mode_idx=config['gmm_data_config']['missing_mode_idx'] # Re-add original missing mode for training set
        )
        image_size_h_w = tuple(config['rasterization_config']['image_size_h_w'])
        train_density_map = rasterize_points_to_density_map(
            gmm_info_training['training_points'],
            image_size_h_w,
            gmm_info_full['data_range_xy'],
            normalize_to=(-1.0, 1.0)
        ).squeeze(0).cpu().numpy()
        full_density_map = rasterize_points_to_density_map(
            gmm_info_full['all_points'],
            image_size_h_w,
            gmm_info_full['data_range_xy'],
            normalize_to=(-1.0, 1.0)
        ).squeeze(0).cpu().numpy()

        wandb_images = {
            "GMM_Density_Map/Training_Set_Map": wandb.Image(train_density_map, caption="Training GMM Density Map (Missing Mode)"),
            "GMM_Density_Map/Full_GMM_Map": wandb.Image(full_density_map, caption="Full GMM Density Map (All Modes)")
        }
        wandb_run.log(wandb_images)


    # --- Perform IBG Sampling ---
    print(f"\n***** Starting IBG Sampling for {config['sampling_config']['num_samples_to_generate']} samples *****")

    latent_shape = (1, model.config.in_channels, image_size_h_w[0], image_size_h_w[1])
    generated_samples_pil = [] # To accumulate PIL images for make_image_grid and W&B

    for i in tqdm(range(config['sampling_config']['num_samples_to_generate']), desc="Generating samples with IBG"):
        sample_generator = torch.Generator(device=device).manual_seed(config['sampling_config']['seed'] + i)

        generated_latent = ibg_sample(
            model=model,
            scheduler=scheduler,
            latent_shape=latent_shape,
            num_inference_steps=config['sampling_config']['num_inference_steps'],
            device=device,
            guidance_scale=config['ibg_config']['guidance_scale'],
            guidance_timesteps_to_apply=config['ibg_config']['guidance_timesteps_to_apply'],
            dis_num_perturbations=config['dis_config']['dis_num_perturbations'],
            dis_perturbation_scale=config['dis_config']['dis_perturbation_scale'],
            generator=sample_generator
        )

        # Post-process generated sample (denormalize and convert to 0-255 uint8)
        # Assuming density maps were normalized to [-1, 1]
        generated_map_np = ((generated_latent / 2 + 0.5).clamp(0, 1) * 255).type(torch.uint8).squeeze(0).squeeze(0).cpu().numpy()

        # Plot and save individual samples
        plot_file_name = f"ibg_sample_{i+1}.png"
        plot_save_path = output_dir / "generated_samples" / plot_file_name

        plot_generated_gmm_sample(
            generated_map_np,
            gmm_info_full,
            i + 1,
            image_size_h_w,
            save_path=plot_save_path,
            dpi=config['evaluation_config']['plot_dpi']
        )

        # Convert to PIL image for make_image_grid and W&B logging
        # The generated_map_np is HxW (uint8). PIL.Image.fromarray expects (H, W) or (H, W, C)
        pil_image = Image.fromarray(generated_map_np)
        generated_samples_pil.append(pil_image)

    # Save a grid of all generated samples
    grid_path = output_dir / "generated_samples" / "all_ibg_samples_grid.png"
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    pil_grid = make_image_grid(generated_samples_pil, rows=1, cols=config['sampling_config']['num_samples_to_generate'])
    pil_grid.save(grid_path)
    print(f"Saved grid of generated samples to {grid_path}")

    if config['evaluation_config']['log_samples_to_wandb']:
        wandb_run.log({"generated_samples_grid": wandb.Image(pil_grid, caption="IBG Generated Samples Grid")})

    print("\nIBG Sampling finished.")
    wandb_run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform IBG sampling on GMM Density Map.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()
    main(args.config)