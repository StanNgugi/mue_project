import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any

# Assuming Diffusers classes are available via their main module,
# though we only use the model (UNet2DModel) and the scheduler (DDPMScheduler)
# directly within the function signature for clarity.
# No direct import of UNet2DModel or DDPMScheduler needed in this file,
# as they are passed as arguments.

def calculate_dis(
    model: torch.nn.Module,
    scheduler: Any, # DDPMScheduler or similar scheduler object
    latents: torch.Tensor, # Noisy latents x_t
    timestep: int,        # Current timestep t
    num_perturbations: int = 10,
    perturbation_scale: float = 0.05, # Standard deviation for Gaussian noise perturbation
    return_raw_dis: bool = False # If True, returns raw sum of squared differences; else, mean.
) -> torch.Tensor:
    """
    Calculates the Denoising Instability Score (DIS) for a given noisy latent
    at a specific timestep.

    DIS quantifies the sensitivity of the model's denoised output to small
    perturbations in its input latent. Higher DIS indicates more uncertainty
    or instability in that region of the manifold.

    Args:
        model (torch.nn.Module): The pre-trained diffusion model (e.g., UNet2DModel).
        scheduler (Any): The diffusion scheduler (e.g., DDPMScheduler) used for
                         denoising steps and converting noise predictions to x_0 estimates.
        latents (torch.Tensor): The current noisy latent tensor (x_t) of shape (B, C, H, W).
                                This function assumes B=1 for simplicity in this minimalist phase.
        timestep (int): The current diffusion timestep t.
        num_perturbations (int): Number of perturbed versions of `latents` to create.
        perturbation_scale (float): Standard deviation of the Gaussian noise added
                                    to `latents` for perturbation.
        return_raw_dis (bool): If True, returns the raw sum of squared differences.
                               If False, returns the mean squared difference.

    Returns:
        torch.Tensor: A spatial map of DIS values of shape (H, W) or (C, H, W)
                      if channels are treated independently.
                      For minimalist GMM, we expect (H, W) for single channel.
    """
    if latents.shape[0] != 1:
        # For simplicity in minimalist phase, we calculate DIS per single latent.
        # Batch DIS calculation would involve broadcasting and more complex aggregation.
        raise ValueError("DIS calculation in this minimalist version expects a batch size of 1.")

    model.eval() # Ensure model is in evaluation mode
    with torch.no_grad(): # No gradients needed for DIS calculation

        # 1. Get the unperturbed model's predicted noise and estimated x_0
        # The model predicts noise (epsilon)
        unperturbed_noise_pred = model(latents, timestep).sample

        # Use scheduler to get the estimated x_0 from the unperturbed noise prediction
        # The `scheduler._get_x_M_from_epsilon` or similar helper is usually internal.
        # We can simulate the x_0 prediction using the common formula for DDPM:
        # x_0_pred = (x_t - sqrt(1 - alpha_prod_t) * epsilon_pred) / sqrt(alpha_prod_t)
        
        # Simulating x_0 estimate using scheduler's internal logic or a common formula:
        # For DDPMScheduler, the model typically predicts noise.
        # We need to compute x_0_pred from x_t and epsilon_pred.
        # This is a common part of the DDPM denoising step.
        alpha_prod_t = scheduler.alphas_cumprod[timestep]
        sqrt_alpha_prod_t = alpha_prod_t ** 0.5
        sqrt_one_minus_alpha_prod_t = (1 - alpha_prod_t) ** 0.5
        
        # Equivalent to scheduler.predict_original_sample(latents, timestep, unperturbed_noise_pred)
        # but for simplicity, we directly compute here.
        # Note: Some schedulers might have slight variations. This is typical for DDPM.
        unperturbed_x0_pred = (latents - sqrt_one_minus_alpha_prod_t * unperturbed_noise_pred) / sqrt_alpha_prod_t


        dis_accumulator = torch.zeros_like(unperturbed_x0_pred, device=latents.device)

        for _ in range(num_perturbations):
            # 2. Generate a perturbation
            perturbation = torch.randn_like(latents) * perturbation_scale
            perturbed_latents = latents + perturbation

            # 3. Get the perturbed model's predicted noise and estimated x_0
            perturbed_noise_pred = model(perturbed_latents, timestep).sample
            perturbed_x0_pred = (perturbed_latents - sqrt_one_minus_alpha_prod_t * perturbed_noise_pred) / sqrt_alpha_prod_t

            # 4. Calculate squared difference in x_0 estimates
            # We are interested in spatial differences, so we calculate this element-wise
            squared_diff = (unperturbed_x0_pred - perturbed_x0_pred)**2
            dis_accumulator += squared_diff

        if return_raw_dis:
            # If we want the sum over perturbations
            dis_map = dis_accumulator
        else:
            # Otherwise, return the mean squared difference
            dis_map = dis_accumulator / num_perturbations
        
        # For a single channel image, squeeze the channel dimension
        return dis_map.squeeze(0) # Returns (H, W) from (1, C, H, W) where C=1


if __name__ == '__main__':
    print("Testing calculate_dis function...")
    
    # Mock components for testing
    from diffusers import UNet2DModel, DDPMScheduler

    # Create a dummy UNet2DModel
    dummy_unet_config = {
        "sample_size": (32, 32),
        "in_channels": 1,
        "out_channels": 1,
        "layers_per_block": 1,
        "block_out_channels": (32, 64),
        "down_block_types": ["DownBlock2D", "DownBlock2D"],
        "up_block_types": ["UpBlock2D", "UpBlock2D"],
    }
    mock_model = UNet2DModel(**dummy_unet_config)
    
    # Create a dummy DDPMScheduler
    mock_scheduler = DDPMScheduler(num_train_timesteps=1000)

    # Create a dummy latent tensor (batch_size=1, channels=1, H=32, W=32)
    test_latents = torch.randn(1, 1, 32, 32)
    test_timestep = 500 # Mid-way timestep
    test_device = "cpu" # Test on CPU for simplicity

    mock_model.to(test_device)
    test_latents = test_latents.to(test_device)

    # Test with default parameters
    print("\n--- Test 1: Default parameters ---")
    dis_map_1 = calculate_dis(mock_model, mock_scheduler, test_latents, test_timestep)
    print(f"DIS Map 1 shape: {dis_map_1.shape}")
    print(f"DIS Map 1 (sample values):\n{dis_map_1[0, 0:5, 0:5] if dis_map_1.ndim==3 else dis_map_1[0:5, 0:5]}")
    assert dis_map_1.shape == (32, 32), "DIS map shape mismatch for Test 1"
    assert dis_map_1.min() >= 0, "DIS values should be non-negative"

    # Test with different num_perturbations and return_raw_dis
    print("\n--- Test 2: num_perturbations=5, return_raw_dis=True ---")
    dis_map_2 = calculate_dis(mock_model, mock_scheduler, test_latents, test_timestep, 
                              num_perturbations=5, return_raw_dis=True)
    print(f"DIS Map 2 shape: {dis_map_2.shape}")
    assert dis_map_2.shape == (32, 32), "DIS map shape mismatch for Test 2"
    assert dis_map_2.min() >= 0, "DIS values should be non-negative"

    # Test with a smaller perturbation scale
    print("\n--- Test 3: perturbation_scale=0.01 ---")
    dis_map_3 = calculate_dis(mock_model, mock_scheduler, test_latents, test_timestep, 
                              perturbation_scale=0.01)
    print(f"DIS Map 3 shape: {dis_map_3.shape}")
    assert dis_map_3.shape == (32, 32), "DIS map shape mismatch for Test 3"

    print("\nDIS calculation tests passed!")

    # Optional: Visualize a DIS map (requires matplotlib)
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 6))
        # For single channel (H, W) DIS map, directly imshow
        plt.imshow(dis_map_1.cpu().numpy(), cmap='hot', origin='lower')
        plt.colorbar(label='DIS Value')
        plt.title('Example DIS Map')
        plt.xlabel('X pixel'); plt.ylabel('Y pixel')
        print("\nPlotted example DIS map. If running headless, plot may not show.")
        # plt.show()
    except ImportError:
        print("\nMatplotlib not installed. Skipping DIS map visualization.")
    except Exception as e:
        print(f"\nError during DIS visualization: {e}")