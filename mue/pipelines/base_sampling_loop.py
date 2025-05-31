import torch
from typing import Callable, Optional, Tuple, Dict, Any
from diffusers import UNet2DModel, DDPMScheduler # Only for type hinting and example usage
from tqdm.auto import tqdm

def ddpm_sample(
    model: UNet2DModel, # Type hint for the model
    scheduler: DDPMScheduler, # Type hint for the scheduler
    latent_shape: Tuple[int, int, int, int], # (batch_size, num_channels, height, width)
    num_inference_steps: int,
    device: torch.device,
    callback_fn: Optional[Callable[[int, torch.Tensor, Dict[str, Any]], None]] = None,
    # callback_fn signature: (timestep, current_latent, model_output_dict)
    # model_output_dict could contain {sample: predicted_noise, ...}
    generator: Optional[torch.Generator] = None # For reproducible sampling
) -> torch.Tensor:
    """
    Performs DDPM sampling to generate images from noise.

    This function provides a basic DDPM sampling loop that can be extended
    or augmented with a callback for custom logic (e.g., DIS calculation, guidance).

    Args:
        model (UNet2DModel): The pre-trained diffusion model.
        scheduler (DDPMScheduler): The diffusion scheduler.
        latent_shape (Tuple[int, int, int, int]): The shape of the initial latents
                                                 (batch_size, num_channels, height, width).
        num_inference_steps (int): The number of denoising steps.
        device (torch.device): The device to run inference on ('cuda' or 'cpu').
        callback_fn (Optional[Callable]): A function to call at each timestep.
                                          Signature: callback_fn(timestep, current_latent, model_output).
                                          `model_output` is the object returned by `model(latents, t)`.
        generator (Optional[torch.Generator]): A torch.Generator object for reproducible sampling.

    Returns:
        torch.Tensor: The final denoised (generated) image tensor of shape (B, C, H, W).
    """
    model.eval() # Set model to evaluation mode
    
    # Initialize latents with random noise
    latents = torch.randn(latent_shape, generator=generator, device=device, dtype=model.dtype)

    # Set the timesteps for the scheduler
    scheduler.set_timesteps(num_inference_steps)

    # Iterate through reversed timesteps for denoising
    for i, t in enumerate(tqdm(scheduler.timesteps, desc="DDPM Sampling")):
        with torch.no_grad(): # No gradients needed during inference
            # 1. Predict noise residual
            # Model output typically contains 'sample' which is the predicted noise
            model_output = model(latents, t).sample
            
            # 2. Optionally, call the callback function
            if callback_fn is not None:
                # Pass the raw model output for flexibility
                callback_fn(t.item(), latents.detach().clone(), {"sample": model_output.detach().clone()})

            # 3. Compute previous noisy sample x_t-1
            # The scheduler.step returns a dict containing 'prev_sample' (x_t-1)
            # and potentially 'pred_original_sample' (x_0 estimate).
            scheduler_output = scheduler.step(model_output, t, latents)
            latents = scheduler_output.prev_sample
            
    # The final latents should be the generated images (x_0)
    return latents


if __name__ == '__main__':
    print("Testing ddpm_sample function...")

    # Mock components for testing
    from diffusers import UNet2DModel, DDPMScheduler

    # Dummy UNet2DModel (very small for quick test)
    mock_unet_config = {
        "sample_size": (32, 32),
        "in_channels": 1,
        "out_channels": 1,
        "layers_per_block": 1,
        "block_out_channels": (16, 16),
        "down_block_types": ["DownBlock2D", "DownBlock2D"],
        "up_block_types": ["UpBlock2D", "UpBlock2D"],
    }
    mock_model = UNet2DModel(**mock_unet_config)
    
    # Dummy DDPMScheduler
    mock_scheduler = DDPMScheduler(num_train_timesteps=100) # Fewer steps for faster test

    test_latent_shape = (1, 1, 32, 32) # Batch size 1
    test_num_inference_steps = 50 # Half the timesteps for quick demo
    test_device = torch.device("cpu") # Test on CPU

    mock_model.to(test_device)

    # --- Test 1: Basic sampling without callback ---
    print("\n--- Test 1: Basic sampling ---")
    generated_image_1 = ddpm_sample(
        model=mock_model,
        scheduler=mock_scheduler,
        latent_shape=test_latent_shape,
        num_inference_steps=test_num_inference_steps,
        device=test_device
    )
    print(f"Generated image 1 shape: {generated_image_1.shape}")
    assert generated_image_1.shape == test_latent_shape, "Generated image shape mismatch for Test 1"
    print(f"Generated image 1 min/max: {generated_image_1.min():.2f}/{generated_image_1.max():.2f}")

    # --- Test 2: Sampling with a dummy callback ---
    print("\n--- Test 2: Sampling with callback ---")
    # Simple callback function to print info
    def simple_callback(timestep_val, current_latent_val, model_output_val):
        print(f"  Callback fired at timestep: {timestep_val}, latent_min: {current_latent_val.min().item():.2f}")
        # In a real scenario, you'd calculate DIS here or log values.

    generated_image_2 = ddpm_sample(
        model=mock_model,
        scheduler=mock_scheduler,
        latent_shape=test_latent_shape,
        num_inference_steps=test_num_inference_steps,
        device=test_device,
        callback_fn=simple_callback
    )
    print(f"Generated image 2 shape: {generated_image_2.shape}")
    assert generated_image_2.shape == test_latent_shape, "Generated image shape mismatch for Test 2"

    # --- Test 3: Reproducible sampling with generator ---
    print("\n--- Test 3: Reproducible sampling ---")
    fixed_seed_generator = torch.Generator(device=test_device).manual_seed(42)
    generated_image_3a = ddpm_sample(
        model=mock_model,
        scheduler=mock_scheduler,
        latent_shape=test_latent_shape,
        num_inference_steps=test_num_inference_steps,
        device=test_device,
        generator=fixed_seed_generator
    )
    fixed_seed_generator = torch.Generator(device=test_device).manual_seed(42) # Re-seed
    generated_image_3b = ddpm_sample(
        model=mock_model,
        scheduler=mock_scheduler,
        latent_shape=test_latent_shape,
        num_inference_steps=test_num_inference_steps,
        device=test_device,
        generator=fixed_seed_generator
    )
    # Check if the images are (almost) identical
    assert torch.allclose(generated_image_3a, generated_image_3b, atol=1e-6), "Reproducible sampling failed"
    print("Reproducible sampling test passed (images are identical).")

    print("\nDDPM sampling function tests passed!")