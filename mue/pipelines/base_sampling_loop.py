import torch
import random
import numpy as np # Import numpy for seeding
from typing import Callable, Optional, Tuple, Dict, Any
from diffusers import UNet2DModel, DDPMScheduler
from tqdm.auto import tqdm

# --- New seeding utility function ---
def set_test_seeds(seed: int):
    """Sets seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic operations (can sometimes slow down or error with certain ops)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True) # Only enable if absolutely necessary and tested

def ddpm_sample(
    model: UNet2DModel,
    scheduler: DDPMScheduler,
    latent_shape: Tuple[int, int, int, int],
    num_inference_steps: int,
    device: torch.device,
    callback_fn: Optional[Callable[[int, torch.Tensor, Dict[str, Any]], None]] = None,
    generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """
    Performs DDPM sampling to generate images from noise.
    """
    model.eval()
    
    latents = torch.randn(latent_shape, generator=generator, device=device, dtype=model.dtype)

    scheduler.set_timesteps(num_inference_steps)

    for i, t in enumerate(tqdm(scheduler.timesteps, desc="DDPM Sampling")):
        with torch.no_grad():
            model_output = model(latents, t).sample
            
            if callback_fn is not None:
                callback_fn(t.item(), latents.detach().clone(), {"sample": model_output.detach().clone()})

            scheduler_output = scheduler.step(model_output, t, latents)
            latents = scheduler_output.prev_sample
            
    return latents


if __name__ == '__main__':
    print("Testing ddpm_sample function...")

    from diffusers import UNet2DModel, DDPMScheduler

    mock_unet_config = {
        "sample_size": (32, 32),
        "in_channels": 1,
        "out_channels": 1,
        "layers_per_block": 1,
        "block_out_channels": (16, 16),
        "down_block_types": ["DownBlock2D", "DownBlock2D"],
        "up_block_types": ["UpBlock2D", "UpBlock2D"],
        "norm_num_groups": 8,
    }
    mock_model = UNet2DModel(**mock_unet_config)
    
    mock_scheduler = DDPMScheduler(num_train_timesteps=100)

    test_latent_shape = (1, 1, 32, 32)
    test_num_inference_steps = 50
    test_device = torch.device("cpu")

    mock_model.to(test_device)

    # --- Test 1: Basic sampling without callback ---
    print("\n--- Test 1: Basic sampling ---")
    # Set seed specifically for this test to ensure its internal operations are deterministic
    set_test_seeds(42) 
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
    def simple_callback(timestep_val, current_latent_val, model_output_val):
        # The tqdm bar might interfere with immediate print, but it works
        pass 
        # print(f"  Callback fired at timestep: {timestep_val}, latent_min: {current_latent_val.min().item():.2f}")

    set_test_seeds(43) # Use a different seed for a distinct test
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
    # Crucially, set the global seed *before* each call for reproducibility
    set_test_seeds(42) 
    fixed_seed_generator_a = torch.Generator(device=test_device).manual_seed(42)
    generated_image_3a = ddpm_sample(
        model=mock_model,
        scheduler=mock_scheduler,
        latent_shape=test_latent_shape,
        num_inference_steps=test_num_inference_steps,
        device=test_device,
        generator=fixed_seed_generator_a # Pass the explicit generator
    )
    
    set_test_seeds(42) # Re-seed the *global* state before the second run
    fixed_seed_generator_b = torch.Generator(device=test_device).manual_seed(42) # Re-seed the generator too
    generated_image_3b = ddpm_sample(
        model=mock_model,
        scheduler=mock_scheduler,
        latent_shape=test_latent_shape,
        num_inference_steps=test_num_inference_steps,
        device=test_device,
        generator=fixed_seed_generator_b # Pass the explicit generator
    )
    
    assert torch.allclose(generated_image_3a, generated_image_3b, atol=1e-6), "Reproducible sampling failed"
    print("Reproducible sampling test passed (images are identical).")

    print("\nDDPM sampling function tests passed!")