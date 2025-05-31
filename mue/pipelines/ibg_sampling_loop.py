import torch
import torch.nn.functional as F
from typing import Callable, Optional, Tuple, Dict, Any, List
from diffusers import UNet2DModel, DDPMScheduler
from tqdm.auto import tqdm

# --- Helper function for differentiable DIS calculation ---
# This is a specialized version of DIS calculation for IBG,
# designed to retain gradients for x_t.
def _compute_differentiable_dis_loss(
    model: UNet2DModel,
    scheduler: DDPMScheduler,
    latents: torch.Tensor, # Noisy latents x_t (requires_grad should be True for this)
    timestep: int,        # Current timestep t
    num_perturbations: int,
    perturbation_scale: float
) -> torch.Tensor:
    """
    Computes a scalar DIS loss that can be backpropagated for guidance.
    This function expects `latents` to have `requires_grad=True`.
    """
    
    # Calculate relevant scheduler parameters for x_0 prediction
    alpha_prod_t = scheduler.alphas_cumprod[timestep]
    sqrt_alpha_prod_t = alpha_prod_t ** 0.5
    sqrt_one_minus_alpha_prod_t = (1 - alpha_prod_t) ** 0.5

    # 1. Get the unperturbed model's predicted noise
    unperturbed_noise_pred = model(latents, timestep).sample
    unperturbed_x0_pred = (latents - sqrt_one_minus_alpha_prod_t * unperturbed_noise_pred) / sqrt_alpha_prod_t

    total_squared_diff = torch.zeros_like(unperturbed_x0_pred) # Keep dimensions for spatial consistency first

    # We average the squared difference over multiple perturbations
    for _ in range(num_perturbations):
        # Generate a perturbation (requires no_grad as it's just input noise)
        perturbation = torch.randn_like(latents) * perturbation_scale
        
        # Perturb the latent (this operation preserves gradients for `latents` if it had them)
        # However, to avoid backpropagating through perturbation generation itself,
        # we detach perturbation if it somehow had a grad history, though it shouldn't here.
        perturbed_latents = latents + perturbation.detach() 

        # Get the perturbed model's predicted noise
        # This model call *must* be in the computation graph for `latents` to get gradients later.
        perturbed_noise_pred = model(perturbed_latents, timestep).sample
        perturbed_x0_pred = (perturbed_latents - sqrt_one_minus_alpha_prod_t * perturbed_noise_pred) / sqrt_alpha_prod_t

        # Calculate squared difference in x_0 estimates for this perturbation
        squared_diff = (unperturbed_x0_pred - perturbed_x0_pred)**2
        total_squared_diff += squared_diff

    # The DIS "loss" is the mean squared difference across all perturbations
    # Sum over spatial dimensions to get a scalar loss for autograd.grad
    # Or, sum over channel/spatial dimensions if we want a single scalar for the batch item.
    # For a spatial gradient for guidance, we actually need the spatial DIS map's gradient.
    # Let's return the mean spatial map first, then sum for gradient calculation.
    mean_spatial_dis_map = total_squared_diff / num_perturbations
    
    # To get a scalar for torch.autograd.grad, sum over all dimensions
    # For a batch of 1 (our current case), this will be (1, 1, H, W) -> scalar
    dis_scalar_loss = mean_spatial_dis_map.mean() # or .sum()

    return dis_scalar_loss # This scalar will be used for gradient computation

def ibg_sample(
    model: UNet2DModel,
    scheduler: DDPMScheduler,
    latent_shape: Tuple[int, int, int, int], # (batch_size, num_channels, height, width)
    num_inference_steps: int,
    device: torch.device,
    guidance_scale: float,
    guidance_timesteps_to_apply: List[int], # List of timesteps where guidance is active
    dis_num_perturbations: int,
    dis_perturbation_scale: float,
    generator: Optional[torch.Generator] = None,
    callback_fn: Optional[Callable[[int, torch.Tensor, Dict[str, Any]], None]] = None, # For optional logging/monitoring
) -> torch.Tensor:
    """
    Performs DDPM sampling with Instability-Biased Guidance (IBG).

    IBG guides the diffusion process towards regions of higher denoising instability
    (i.e., less confident regions) to encourage exploration of unlearned parts of the manifold.

    Args:
        model (UNet2DModel): The pre-trained diffusion model.
        scheduler (DDPMScheduler): The diffusion scheduler.
        latent_shape (Tuple[int, int, int, int]): The shape of the initial latents.
        num_inference_steps (int): The number of denoising steps.
        device (torch.device): The device to run inference on.
        guidance_scale (float): Strength of the instability bias.
        guidance_timesteps_to_apply (List[int]): List of timesteps where IBG will be active.
        dis_num_perturbations (int): Number of perturbations for DIS gradient calculation.
        dis_perturbation_scale (float): Scale of perturbations for DIS gradient calculation.
        generator (Optional[torch.Generator]): A torch.Generator for reproducible sampling.
        callback_fn (Optional[Callable]): An optional callback function to monitor process.

    Returns:
        torch.Tensor: The final denoised (generated) image tensor.
    """
    model.eval() # Set model to evaluation mode

    latents = torch.randn(latent_shape, generator=generator, device=device, dtype=model.dtype)
    scheduler.set_timesteps(num_inference_steps)

    # Convert guidance_timesteps_to_apply to a set for faster lookup
    guidance_timesteps_set = set(guidance_timesteps_to_apply)

    for i, t in enumerate(tqdm(scheduler.timesteps, desc="IBG Sampling")):
        # Store initial latents for gradient computation if guidance is applied
        # We need to clone and detach to ensure we don't accidentally compute gradients
        # for previous steps if we didn't intend to.
        # However, for autograd.grad, the `latents` must be part of the graph.
        
        with torch.no_grad():
            # Get the unconditional noise prediction first (no_grad for base prediction)
            # This is equivalent to epsilon_theta(x_t, t)
            uncond_noise_pred = model(latents, t).sample
        
        guided_noise_pred = uncond_noise_pred # Start with unconditional prediction

        # Apply IBG if current timestep is in the active guidance steps
        if t.item() in guidance_timesteps_set:
            # Prepare latents for gradient computation
            latents.requires_grad_(True) 
            
            # Compute the differentiable DIS loss
            # This function re-runs model calls and builds a graph from `latents`
            dis_loss = _compute_differentiable_dis_loss(
                model=model,
                scheduler=scheduler,
                latents=latents,
                timestep=t,
                num_perturbations=dis_num_perturbations,
                perturbation_scale=dis_perturbation_scale
            )
            
            # Compute the gradient of the DIS loss with respect to the latents (x_t)
            # This gives us the direction to move in latent space to increase DIS.
            # grad_outputs=torch.ones_like(dis_loss) is implicitly handled when output is scalar
            grad_dis = torch.autograd.grad(outputs=dis_loss, inputs=latents)[0]

            # The guidance term for noise prediction is proportional to -guidance_scale * grad_dis
            # The scaling factor: In DDPM, the score function is related to -epsilon / sigma_t.
            # So, to add a guidance term (w * grad_F) to the score, we modify epsilon by -w * sigma_t * grad_F.
            # Here, sigma_t = sqrt(1 - alpha_prod_t)
            # The term `sqrt_one_minus_alpha_prod_t` should be from the scheduler's internal beta schedule.
            
            # Using scheduler's internal sqrt_one_minus_alpha_prod to scale the gradient
            # This is common in many guidance implementations (e.g., Classifier Guidance)
            # The scheduler's alpha_prod_t is alpha_t_bar
            sqrt_one_minus_alpha_prod_t = scheduler.alphas_cumprod[t] ** 0.5 # this is sqrt(alpha_bar_t), not sqrt(1-alpha_bar_t)
            # No, standard guidance typically uses std_dev of noise: sqrt(1 - alpha_prod_t)
            # Let's re-verify the correct scaling factor for gradient guidance in epsilon space.
            # From common guidance formulas (e.g., from DDIM, classifier guidance):
            # epsilon_guided = epsilon_uncond - w * sqrt(1 - alpha_prod_t) * grad_log_p_data
            # For our case, we want to maximize DIS, so we're adding -w * grad_DIS to the score.
            # So epsilon_guided = epsilon_uncond - w * sqrt(1 - alpha_prod_t) * grad_DIS_wrt_latent.
            
            # Retrieve parameters for scaling from scheduler.
            # Note: scheduler.alphas_cumprod is alpha_bar_t
            # scheduler.sqrt_one_minus_alphas_cumprod is sqrt(1 - alpha_bar_t) (the std dev of noise)
            
            # Correct scaling factor for DDPM-like guidance: std_dev_t = sqrt(1 - alpha_prod_t)
            std_dev_t = scheduler.sqrt_one_minus_alphas_cumprod[t] if hasattr(scheduler, 'sqrt_one_minus_alphas_cumprod') else (1 - scheduler.alphas_cumprod[t])**0.5
            
            # Apply guidance to the noise prediction
            # The gradient points in the direction of *increasing* DIS.
            # To push towards higher DIS, we add a term proportional to `grad_dis` to `epsilon_pred`.
            # This might seem counter-intuitive as epsilon is noise, but in score-based models,
            # score is proportional to -epsilon. So, -score_guided = -score_uncond + w * grad_F.
            # -> epsilon_guided = epsilon_uncond - w * sigma_t * grad_F.
            
            guided_noise_pred = uncond_noise_pred - guidance_scale * std_dev_t * grad_dis
            
            # After computing gradient and applying guidance, detach latents for next step
            latents = latents.detach()
            
        # Optional callback for monitoring
        if callback_fn is not None:
            callback_fn(t.item(), latents.detach().clone(), {"sample": guided_noise_pred.detach().clone()})

        # Compute previous noisy sample x_t-1 using the (potentially) guided noise prediction
        scheduler_output = scheduler.step(guided_noise_pred, t, latents)
        latents = scheduler_output.prev_sample
            
    return latents

if __name__ == '__main__':
    print("Testing ibg_sample function...")

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
    test_num_inference_steps = 50 
    test_device = torch.device("cpu") # Test on CPU

    mock_model.to(test_device)

    # --- Test 1: IBG sampling with specific parameters ---
    print("\n--- Test 1: Basic IBG sampling ---")
    test_guidance_scale = 1.0
    # Apply guidance only at a few high timesteps for this test
    test_guidance_timesteps = [90, 80, 70] 
    test_dis_num_perturbations = 3 # Small for quick test
    test_dis_perturbation_scale = 0.01

    generated_image = ibg_sample(
        model=mock_model,
        scheduler=mock_scheduler,
        latent_shape=test_latent_shape,
        num_inference_steps=test_num_inference_steps,
        device=test_device,
        guidance_scale=test_guidance_scale,
        guidance_timesteps_to_apply=test_guidance_timesteps,
        dis_num_perturbations=test_dis_num_perturbations,
        dis_perturbation_scale=test_dis_perturbation_scale
    )
    print(f"Generated image shape: {generated_image.shape}")
    assert generated_image.shape == test_latent_shape, "Generated image shape mismatch"
    print(f"Generated image min/max: {generated_image.min():.2f}/{generated_image.max():.2f}")

    # --- Test 2: Check if guidance actually modifies prediction (simple qualitative check) ---
    # This is a conceptual check, not a strict assert
    print("\n--- Test 2: Qualitative check for guidance effect ---")
    # Generate without guidance
    no_guidance_image = ibg_sample(
        model=mock_model,
        scheduler=mock_scheduler,
        latent_shape=test_latent_shape,
        num_inference_steps=test_num_inference_steps,
        device=test_device,
        guidance_scale=0.0, # No guidance
        guidance_timesteps_to_apply=test_guidance_timesteps, # Ignored if guidance_scale is 0
        dis_num_perturbations=test_dis_num_perturbations,
        dis_perturbation_scale=test_dis_perturbation_scale,
        generator=torch.Generator(device=test_device).manual_seed(0) # Fixed seed
    )

    # Generate with guidance
    with_guidance_image = ibg_sample(
        model=mock_model,
        scheduler=mock_scheduler,
        latent_shape=test_latent_shape,
        num_inference_steps=test_num_inference_steps,
        device=test_device,
        guidance_scale=test_guidance_scale, # With guidance
        guidance_timesteps_to_apply=test_guidance_timesteps,
        dis_num_perturbations=test_dis_num_perturbations,
        dis_perturbation_scale=test_dis_perturbation_scale,
        generator=torch.Generator(device=test_device).manual_seed(0) # Same fixed seed
    )
    
    # These should be different if guidance has an effect
    difference = (no_guidance_image - with_guidance_image).abs().sum()
    print(f"Absolute difference between no-guidance and with-guidance images: {difference.item():.2f}")
    assert difference > 1e-3, "Guidance likely had no effect (difference too small)"

    print("\nIBG sampling function tests passed!")