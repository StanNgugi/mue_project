import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, List, Dict, Any, Optional

# Assuming rasterize_points_to_density_map is in the same directory or accessible via PYTHONPATH
from .rasterizers import rasterize_points_to_density_map 
from .synthetic_generators import get_gmm_data_for_training_and_evaluation


class Synthetic2DDensityDataset(Dataset):
    """
    PyTorch Dataset for serving rasterized 2D synthetic point data as density maps.
    """
    def __init__(self,
                 points: np.ndarray,
                 image_size: Tuple[int, int], # (H, W)
                 data_range_xy: Tuple[Tuple[float, float], Tuple[float, float]], # ((min_x, max_x), (min_y, max_y))
                 normalize_to: Optional[Tuple[float, float]] = (-1.0, 1.0),
                 pre_rasterize: bool = True):
        """
        Args:
            points (np.ndarray): Array of shape (N_total_points, 2) for (x,y) coordinates.
                                 The dataset will create one "image" per point if not pre_rasterize,
                                 or one image from all points if pre_rasterize is set to make a single image
                                 (which is unusual for typical image datasets, more common for point clouds).
                                 This class assumes each item in the dataset IS a density map generated
                                 from a SUBSET of points, or in our GMM case, one map per GMM *sample*.
                                 For a DDPM on a GMM, we typically generate many "images" where each image
                                 is a density map of *one* sample from the GMM distribution.
                                 OR, if we interpret this differently: each "sample" to the DDPM is a complete
                                 density map of *all* training points. This latter interpretation is less common
                                 for image DDPMs but might be what's intended for learning a distribution's density.

                                 Let's clarify: For DDPMs on images, each item in the dataset is *an image*.
                                 If our "image" is a density map of the GMM, we need *many such images*.
                                 This means we need to generate N GMM point sets, rasterize each.
                                 However, the typical use for synthetic data like GMM for DDPM is to learn the
                                 manifold of the *points themselves*, not an image of their density.
                                 If the UNet expects image-like input, the rasterization approach is common.
                                 Let's assume for now each dataset item is a density map representing ONE sample
                                 from the GMM, but highly localized (like a dirac delta if not careful).

                                 Re-thinking: For a DDPM to learn a GMM distribution and output *samples* that look
                                 like density maps of that GMM, the training data should be density maps *of the GMM itself*.
                                 So, if we have N_train_points from a GMM, we create ONE density map from these N_train_points.
                                 Then, the DDPM is trained on *this single image* (or slightly perturbed versions).
                                 This trains the DDPM to generate *that specific density map*. This seems to be the most direct
                                 interpretation of "training a DDPM on synthetic data like GMM" when the DDPM is image-based.

                                 If `pre_rasterize` is True, it creates one image. If False, it assumes `points` is a list of point sets.
                                 Given the MUE context, we want to train a DDPM on a *distribution*. So we need many samples
                                 from this distribution. If each "sample" is an image, then we need many images.
                                 The current setup seems to imply we have a point cloud, and we want to create *one* density map.
                                 This single density map would then be the *single* training image for the DDPM.

            image_size (Tuple[int, int]): Target image dimensions (H, W).
            data_range_xy (Tuple[Tuple[float, float], Tuple[float, float]]): Overall coordinate range.
            normalize_to (Optional[Tuple[float, float]]): Target range for normalization.
            pre_rasterize (bool): If True, rasterizes all points into a single image during init.
                                  The dataset will then have only one item. This is suitable for training
                                  a DDPM to reconstruct/generate this specific density map.
        """
        self.points = points
        self.image_size = image_size
        self.data_range_xy = data_range_xy
        self.normalize_to = normalize_to
        self.pre_rasterize = pre_rasterize

        if self.pre_rasterize:
            self.rasterized_map = rasterize_points_to_density_map(
                self.points, self.image_size, self.data_range_xy, self.normalize_to
            )
        else:
            # This mode would require points to be a list of point sets, or to be sampled from.
            # For now, focusing on the single density map case for the GMM.
            raise NotImplementedError("On-the-fly rasterization for multiple samples not implemented yet. Use pre_rasterize=True for a single density map.")

    def __len__(self):
        return 1 if self.pre_rasterize else len(self.points) # Or however many "images" we intend to make

    def __getitem__(self, idx):
        if self.pre_rasterize:
            if idx != 0:
                raise IndexError("Dataset configured with pre_rasterize=True only has one item at index 0.")
            return self.rasterized_map
        else:
            # This part would handle generating/fetching the idx-th density map sample.
            raise NotImplementedError()

def get_gmm_dataloader(
    gmm_config: Dict[str, Any], # From get_gmm_data_for_training_and_evaluation['config']
    image_size: Tuple[int, int],
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    use_all_points_for_map: bool = False # If true, uses 'all_points', else 'training_points'
) -> Tuple[DataLoader, Dict[str, Any]]:
    """
    Creates a DataLoader for the GMM density map dataset.
    The dataset will contain a single item: the density map of the specified GMM points.

    Args:
        gmm_config (Dict[str, Any]): Configuration dictionary used to generate GMM data.
                                      Used to regenerate data consistently.
        image_size (Tuple[int, int]): Target image dimensions (H, W) for rasterization.
        batch_size (int): Batch size for the DataLoader. Since dataset has 1 item, this is often 1.
        shuffle (bool): Whether to shuffle (less relevant for single-item dataset).
        num_workers (int): Number of DataLoader workers.
        pin_memory (bool): DataLoader pin_memory.
        use_all_points_for_map (bool): If True, rasterizes all generated GMM points.
                                       If False, rasterizes only the 'training_points' (missing mode excluded).

    Returns:
        Tuple[DataLoader, Dict[str, Any]]: The DataLoader and the full gmm_info dictionary.
    """
    gmm_info = get_gmm_data_for_training_and_evaluation(**gmm_config)
    
    points_to_rasterize = gmm_info['all_points'] if use_all_points_for_map else gmm_info['training_points']
    
    dataset = Synthetic2DDensityDataset(
        points=points_to_rasterize,
        image_size=image_size,
        data_range_xy=gmm_info['data_range_xy'], # Use consistent range from all points
        normalize_to=(-1.0, 1.0),
        pre_rasterize=True # Create one density map image
    )

    # If dataset has only 1 item, batch_size > 1 will still yield batches of size 1.
    # Consider if the DDPM training loop expects batches or single items.
    # For image DDPMs, batch_size is usually > 1. If we train on a single image,
    # the "batch" will be that single image repeated, or the training loop needs to handle it.
    # Often, for training on a single image, one might apply augmentations on-the-fly.
    # Here, we'll provide the single image as is.
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size, # Typically 1 if dataset has 1 item, unless training loop duplicates
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return dataloader, gmm_info


if __name__ == '__main__':
    print("Testing GMM DataLoader...")
    
    config_params = {
        'n_total_modes': 4,
        'n_samples_per_mode': 1000, # More samples for a better density map
        'missing_mode_idx': 0,
        'layout': 'square',
        'distance_scale': 2.0,
        'cluster_std_dev': 0.2,
        'random_state': 42
    }
    img_h_w = (64, 64)
    bs = 1 # Batch size for a dataset that will contain 1 image

    # Create DataLoader for the training density map (missing one mode)
    train_dataloader, train_gmm_info = get_gmm_dataloader(
        gmm_config=config_params,
        image_size=img_h_w,
        batch_size=bs,
        use_all_points_for_map=False # Use 'training_points'
    )

    print(f"Training DataLoader created. Number of batches: {len(train_dataloader)}")
    train_density_map_batch = next(iter(train_dataloader))
    print(f"  Batch shape: {train_density_map_batch.shape}") # Should be (bs, 1, H, W)
    print(f"  Min value in training map batch: {train_density_map_batch.min().item():.2f}")
    print(f"  Max value in training map batch: {train_density_map_batch.max().item():.2f}")
    assert train_density_map_batch.shape == (bs, 1, img_h_w[0], img_h_w[1])

    # Create a density map of ALL points for reference (e.g., for visualization or FID reference if needed)
    # This is not typically used for training directly unless missing_mode_idx is None
    full_dataloader, full_gmm_info = get_gmm_dataloader(
        gmm_config=config_params,
        image_size=img_h_w,
        batch_size=bs,
        use_all_points_for_map=True # Use 'all_points'
    )
    full_density_map_batch = next(iter(full_dataloader))
    print(f"\nFull GMM DataLoader created (all modes included in map). Number of batches: {len(full_dataloader)}")
    print(f"  Batch shape: {full_density_map_batch.shape}")
    
    print("GMM DataLoader test passed!")

    # Optional: Visualize the density maps from DataLoader
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        im0 = axes[0].imshow(train_density_map_batch.squeeze().cpu().numpy(), cmap='viridis', origin='lower', 
                       extent=[train_gmm_info['data_range_xy'][0][0], train_gmm_info['data_range_xy'][0][1], 
                               train_gmm_info['data_range_xy'][1][0], train_gmm_info['data_range_xy'][1][1]])
        axes[0].set_title('Training GMM Density Map (Missing Mode)')
        axes[0].set_xlabel('X'); axes[0].set_ylabel('Y')
        fig.colorbar(im0, ax=axes[0], label='Normalized Density')

        im1 = axes[1].imshow(full_density_map_batch.squeeze().cpu().numpy(), cmap='viridis', origin='lower',
                       extent=[full_gmm_info['data_range_xy'][0][0], full_gmm_info['data_range_xy'][0][1],
                               full_gmm_info['data_range_xy'][1][0], full_gmm_info['data_range_xy'][1][1]])
        axes[1].set_title('Full GMM Density Map (All Modes)')
        axes[1].set_xlabel('X'); axes[1].set_ylabel('Y')
        fig.colorbar(im1, ax=axes[1], label='Normalized Density')
        
        plt.tight_layout()
        # plt.savefig('gmm_dataloader_density_maps.png')
        print("\nPlotted density maps from DataLoader. If running headless, plot may not show.")
        # plt.show()
    except ImportError:
        print("\nMatplotlib not installed. Skipping DataLoader density map visualization.")
    except Exception as e:
        print(f"\nError during visualization: {e}")