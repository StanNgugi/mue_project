import argparse
import os
import shutil
import yaml
from pathlib import Path
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors # For custom colormaps
from matplotlib.cm import ScalarMappable # For colorbars
from tqdm.auto import tqdm
import wandb

# Import necessary types from 'typing' module
from typing import Dict, Any, Optional, Tuple, Callable 

# Project-specific imports
from mue.data_handling.synthetic_generators import get_gmm_data_for_training_and_evaluation
from mue.data_handling.rasterizers import rasterize_points_to_density_map
from mue.dis.score import calculate_dis
from mue.pipelines.base_sampling_loop import ddpm_sample
from mue.utils.seeding import set_global_seeds
from mue.utils.logging import initialize_wandb # Assuming this utility exists

# Diffusers imports for model and scheduler
from diffusers import UNet2DModel, DDPMScheduler

def load_config(config_path: str) -> dict:
    """Loads a YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_trained_model_and_scheduler(model_load_config: dict, device: torch.device):
    """Loads a trained UNet2DModel and DDPMScheduler."""
    model_path = Path(model_load_config['trained_model_output_dir']) / "checkpoint_final"
    if not model_path.exists():
        # Fallback if 'final' checkpoint doesn't exist, try to load directly from base dir
        print(f"Warning: 'checkpoint_final' not found at {model_path}. Trying to load directly from {model_load_config['trained_model_output_dir']}.")
        model_path = Path(model_load_config['trained_model_output_dir'])

    if not model_path.exists():
        raise FileNotFoundError(f"Trained model checkpoint not found at {model_path}. Please check 'trained_model_output_dir' in config.")

    # Load UNet
    model = UNet2DModel.from_pretrained(str(model_path)).to(device)
    print(f"Loaded UNet model from: {model_path}")

    # Load Scheduler config (scheduler itself is initialized from default params then config loaded)
    scheduler_config_path = model_path / "scheduler_config.json"
    if not scheduler_config_path.exists():
        print(f"Warning: scheduler_config.json not found at {scheduler_config_path}. Initializing DDPMScheduler with default parameters. This might lead to mismatches if training used non-default scheduler params.")
        scheduler = DDPMScheduler()
    else:
        scheduler = DDPMScheduler.from_config(str(scheduler_config_path))
    print(f"Loaded DDPMScheduler config from: {scheduler_config_path if scheduler_config_path.exists() else 'defaults'}")

    return model, scheduler

def plot_dis_map(
    dis_map: np.ndarray,
    gmm_data_info: Dict[str, Any],
    timestep: int,
    image_size: Tuple[int, int],
    save_path: Optional[Path] = None,
    dpi: int = 100):
    """
    Plots the DIS map, overlaid with GMM mode centers and points.
    Highlights the missing mode.
    """
    fig, ax = plt.subplots(figsize=(8, 7), dpi=dpi)

    (min_x, max_x), (min_y, max_y) = gmm_data_info['data_range_xy']
    extent = [min_x, max_x, min_y, max_y]

    # Plot the DIS map
    # Normalize DIS for visualization
    vmax = np.max(dis_map) # Max value of DIS for color normalization
    norm = mcolors.Normalize(vmin=0, vmax=vmax)
    cmap = plt.cm.plasma # 'hot' or 'viridis' or 'plasma'
    
    im = ax.imshow(dis_map, cmap=cmap, norm=norm, origin='lower', extent=extent, alpha=0.9)
    cbar = fig.colorbar(im, ax=ax, label='Mean DIS Value', shrink=0.7)

    # Overlay GMM points (lighter color for training, red for missing)
    # Re-rasterize full GMM for background context if needed, or use scatter directly
    # For clear visualization, scattering the actual points is better.
    
    # All modes' centers
    mode_centers_x = [c[0] for c in gmm_data_info['mode_centers']]
    mode_centers_y = [c[1] for c in gmm_data_info['mode_centers']]
    ax.scatter(mode_centers_x, mode_centers_y, marker='X', color='white', s=200, edgecolor='black', linewidth=1.5, zorder=3, label='All Mode Centers')


    # Scatter plot for all_points, distinguishing training vs missing
    if gmm_data_info['missing_mode_center'] is not None:
        training_points_labels = gmm_data_info['training_labels'].copy()
        # Remap labels to exclude missing_mode_idx if needed for consistent colors
        # For simple scatter, just plot points with original labels
        
        # Plot training points (excluding the missing mode)
        ax.scatter(gmm_data_info['training_points'][:, 0], gmm_data_info['training_points'][:, 1], 
                   c=training_points_labels, cmap='viridis', s=5, alpha=0.3, zorder=1, label='Training Points')
        
        # Plot missing mode points
        if len(gmm_data_info['missing_mode_points']) > 0:
            ax.scatter(gmm_data_info['missing_mode_points'][:, 0], gmm_data_info['missing_mode_points'][:, 1], 
                       color='red', s=10, alpha=0.6, zorder=2, label=f"Missing Mode {gmm_data_info['config']['missing_mode_idx']} Points")
            # Highlight missing mode center with a larger marker
            missing_center = gmm_data_info['missing_mode_center']
            ax.scatter(missing_center[0], missing_center[1], marker='*', color='gold', s=400, edgecolor='black', linewidth=2, zorder=4, label=f"Missing Mode {gmm_data_info['config']['missing_mode_idx']} Center")
    else:
        # If no mode was missing (e.g., full GMM visualization)
        ax.scatter(gmm_data_info['all_points'][:, 0], gmm_data_info['all_points'][:, 1], 
                   c=gmm_data_info['all_labels'], cmap='viridis', s=5, alpha=0.3, zorder=1, label='All GMM Points')


    ax.set_title(f"DIS Map at Timestep {timestep} (Overlay with GMM)")
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='upper right', fontsize='small')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        plt.close(fig) # Close plot to free memory
    else:
        plt.show()

def main(config_path: str):
    config = load_config(config_path)

    # Setup output directory
    output_dir = Path(config['evaluation_config']['output_dir'])
    if output_dir.exists():
        print(f"Warning: Output directory {output_dir} already exists. Contents might be overwritten.")
        # Optionally add a cleanup or unique naming here
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the config file to the output directory for reproducibility
    with open(output_dir / "eval_config.yaml", 'w') as f:
        yaml.dump(config, f)

    # --- Initialize ---
    set_global_seeds(config['sampling_config']['seed'])
    device = torch.device(config['evaluation_config']['device'] if torch.cuda.is_available() else "cpu")
    
    if device.type == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Initialize W&B
    run_name = f"{config['wandb_config']['run_name_prefix']}_seed{config['sampling_config']['seed']}"
    wandb_run = initialize_wandb(
        project_name=config['wandb_config']['project_name'],
        run_name=run_name,
        config=config, # Log the entire config
        entity=config['wandb_config'].get('entity')
    )

    # --- Generate GMM Data for Reference (ensure consistency with training) ---
    # Create a copy of the GMM config for generating the FULL GMM (no missing mode)
    gmm_data_config_for_full_gmm = config['gmm_data_config'].copy()
    gmm_data_config_for_full_gmm['random_state'] = config['sampling_config']['seed'] 
    # Crucially, set 'missing_mode_idx' to None for the full GMM generation BEFORE unpacking
    gmm_data_config_for_full_gmm['missing_mode_idx'] = None 
    
    gmm_info_full = get_gmm_data_for_training_and_evaluation(
        **gmm_data_config_for_full_gmm # Now, this correctly passes missing_mode_idx=None
    )
    print(f"Reference GMM data range: {gmm_info_full['data_range_xy']}")
    
    # Create another copy of the GMM config for generating the TRAINING GMM (with missing mode)
    gmm_data_config_for_training = config['gmm_data_config'].copy()
    gmm_data_config_for_training['random_state'] = config['sampling_config']['seed'] 
    # This copy retains the original missing_mode_idx from the config
    
    gmm_info_training = get_gmm_data_for_training_and_evaluation(
        **gmm_data_config_for_training # This correctly passes the configured missing_mode_idx
    )

    # --- Load Trained Model and Scheduler ---
    model, scheduler = load_trained_model_and_scheduler(config['model_load_config'], device)
    
    # Rasterize the training density map and full density map for initial logging
    image_size_h_w = tuple(config['rasterization_config']['image_size_h_w'])
    train_density_map = rasterize_points_to_density_map(
        gmm_info_training['training_points'], 
        image_size_h_w, 
        gmm_info_full['data_range_xy'], # Use full range for consistency
        normalize_to=(-1.0, 1.0)
    ).squeeze(0).cpu().numpy()

    full_density_map = rasterize_points_to_density_map(
        gmm_info_full['all_points'], 
        image_size_h_w, 
        gmm_info_full['data_range_xy'], # Use full range for consistency
        normalize_to=(-1.0, 1.0)
    ).squeeze(0).cpu().numpy()

    # Log initial maps to W&B
    if config['evaluation_config']['log_gmm_info_to_wandb']:
        wandb_images = {
            "GMM_Density_Map/Training_Set_Map": wandb.Image(train_density_map, caption="Training GMM Density Map (Missing Mode)"),
            "GMM_Density_Map/Full_GMM_Map": wandb.Image(full_density_map, caption="Full GMM Density Map (All Modes)")
        }
        wandb_run.log(wandb_images)


    # --- DIS Calculation Setup ---
    dis_timesteps_to_evaluate = sorted(list(config['dis_config']['dis_timesteps_to_evaluate']), reverse=True)
    
    # Accumulators for DIS maps across multiple generations
    # Store sum of DIS maps, then divide by num_generations_for_dis_avg at the end
    dis_accumulators = {t: torch.zeros(image_size_h_w, device=device) for t in dis_timesteps_to_evaluate}
    
    # Define the callback function for ddpm_sample
    def dis_eval_callback(timestep_val: int, current_latent: torch.Tensor, model_output: Dict[str, Any]):
        if timestep_val in dis_timesteps_to_evaluate:
            # We already have model_output['sample'] (predicted noise), but calculate_dis
            # needs the actual model and scheduler to re-evaluate x_0 predictions from perturbations.
            
            # Ensure the current_latent is a single item batch (calculate_dis expects B=1)
            # If ddpm_sample were generating a batch, this would need to be handled.
            # But in base_sampling_loop, latent_shape is (1, C, H, W), so current_latent is (1, C, H, W).
            
            # Calculate DIS for this latent and timestep
            dis_map_for_step = calculate_dis(
                model=model,
                scheduler=scheduler,
                latents=current_latent,
                timestep=timestep_val,
                num_perturbations=config['dis_config']['dis_num_perturbations'],
                perturbation_scale=config['dis_config']['dis_perturbation_scale'],
                return_raw_dis=False # We want the mean squared difference for averaging later
            )
            dis_accumulators[timestep_val] += dis_map_for_step # Accumulate sum


    # --- Run Sampling and DIS Calculation ---
    print(f"\n***** Starting DIS Evaluation over {config['sampling_config']['num_generations_for_dis_avg']} generations *****")
    
    latent_shape = (1, model.config.in_channels, image_size_h_w[0], image_size_h_w[1])
    inference_generator = torch.Generator(device=device) # Generator for each run

    for i in tqdm(range(config['sampling_config']['num_generations_for_dis_avg']), desc="Running generations for DIS"):
        # Set new seed for each generation for independent runs
        inference_generator.manual_seed(config['sampling_config']['seed'] + i) 
        
        # Run the sampling process. DIS calculation happens via the callback.
        _ = ddpm_sample(
            model=model,
            scheduler=scheduler,
            latent_shape=latent_shape,
            num_inference_steps=config['sampling_config']['num_inference_steps'],
            device=device,
            callback_fn=dis_eval_callback,
            generator=inference_generator
        )

    # --- Process and Visualize Averaged DIS Maps ---
    print("\nProcessing and plotting averaged DIS maps...")
    
    plots_dir = output_dir / "dis_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    wandb_dis_images = {}

    for timestep in dis_timesteps_to_evaluate:
        avg_dis_map_tensor = dis_accumulators[timestep] / config['sampling_config']['num_generations_for_dis_avg']
        avg_dis_map_np = avg_dis_map_tensor.cpu().numpy()

        plot_file_name = f"dis_map_t{timestep}.png"
        plot_save_path = plots_dir / plot_file_name

        print(f"Plotting DIS map for timestep: {timestep}")
        plot_dis_map(
            avg_dis_map_np,
            gmm_info_full, # Use full GMM info for overlay reference
            timestep,
            image_size_h_w,
            save_path=plot_save_path,
            dpi=config['evaluation_config']['plot_dpi']
        )
        
        if config['evaluation_config']['log_dis_plots_to_wandb']:
            wandb_dis_images[f"DIS_Maps/t_{timestep}"] = wandb.Image(str(plot_save_path), caption=f"Avg DIS Map at Timestep {timestep}")
    
    if config['evaluation_config']['log_dis_plots_to_wandb']:
        wandb_run.log(wandb_dis_images)

    print("\nDIS Evaluation finished.")
    wandb_run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DIS on a GMM Density Map.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file for DIS evaluation.")
    args = parser.parse_args()
    main(args.config)