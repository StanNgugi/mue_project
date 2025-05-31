import numpy as np
import torch
from typing import Tuple, Optional

def rasterize_points_to_density_map(
    points: np.ndarray,
    image_size: Tuple[int, int], # (H, W)
    data_range_xy: Tuple[Tuple[float, float], Tuple[float, float]], # ((min_x, max_x), (min_y, max_y))
    normalize_to: Optional[Tuple[float, float]] = (-1.0, 1.0) # Target range e.g. [-1,1] or [0,1]
) -> torch.Tensor:
    """
    Rasterizes 2D points into a 2D density map (histogram) and normalizes it.

    Args:
        points (np.ndarray): Array of shape (N, 2) of (x, y) coordinates.
                             If empty, returns a zero tensor.
        image_size (Tuple[int, int]): Target image dimensions (H, W).
        data_range_xy (Tuple[Tuple[float, float], Tuple[float, float]]):
            The ((min_x, max_x), (min_y, max_y)) range of the coordinate space
            to be mapped onto the image. Points outside this range will be clipped by np.histogram2d.
        normalize_to (Optional[Tuple[float, float]]): If provided, scales the density map
            to this range [min_val, max_val]. If None, raw counts are returned.

    Returns:
        torch.Tensor: A tensor of shape (1, H, W) representing the density map.
    """
    H, W = image_size
    (min_x, max_x), (min_y, max_y) = data_range_xy

    if points.shape[0] == 0: # Handle empty input points
        density_map = np.zeros((H, W), dtype=np.float32)
    else:
        # Ensure points are within the specified range for histogramming
        # np.histogram2d handles points outside the range by not counting them or counting in under/overflow bins if not strict.
        # For our purpose, the range defines the canvas.

        # Bins: W for x-dimension, H for y-dimension for np.histogram2d
        # The output histogram H_np has shape (nx, ny) so hist[i,j] is count for x_bins[i]<=x<x_bins[i+1], y_bins[j]<=y<y_bins[j+1]
        # We want our image to be (H, W) where H is rows (y-axis), W is columns (x-axis)
        hist, x_edges, y_edges = np.histogram2d(
            points[:, 0], points[:, 1],
            bins=[W, H],  # W bins for x, H bins for y
            range=[[min_x, max_x], [min_y, max_y]],
            density=False # Get counts, not probability density
        )
        # hist from np.histogram2d has x-bins along the first dimension, y-bins along the second.
        # To match image convention (Height first, Width second), we need to transpose it.
        density_map = hist.T.astype(np.float32) # Transpose to get (H, W)

    if normalize_to is not None:
        min_val_target, max_val_target = normalize_to
        map_min, map_max = density_map.min(), density_map.max()
        if map_max > map_min: # Avoid division by zero if map is flat
            density_map = (density_map - map_min) / (map_max - map_min) # Scale to [0, 1]
            density_map = density_map * (max_val_target - min_val_target) + min_val_target # Scale to target range
        elif map_max == map_min and map_min != min_val_target : # Flat map, not already at target min
             # If all values are same, set to min_val_target or middle of target range
             density_map.fill(min_val_target if min_val_target == max_val_target else (min_val_target + max_val_target) / 2)


    return torch.from_numpy(density_map).unsqueeze(0) # Add channel dimension: (1, H, W)


if __name__ == '__main__':
    # This is a placeholder for `synthetic_generators`.
    # In a real scenario, you would have this module or replace it with your data loading.
    class SyntheticGenerators:
        def get_gmm_data_for_training_and_evaluation(self, n_samples_per_mode, random_state, layout, distance_scale, cluster_std_dev):
            # For demonstration, we'll create some dummy data
            np.random.seed(random_state)
            if layout == 'square':
                # Create a simple 2-mode GMM-like structure
                mean1 = np.array([-0.5, -0.5]) * distance_scale
                mean2 = np.array([0.5, 0.5]) * distance_scale
                cov = np.array([[cluster_std_dev**2, 0], [0, cluster_std_dev**2]])

                points1 = np.random.multivariate_normal(mean1, cov, n_samples_per_mode)
                points2 = np.random.multivariate_normal(mean2, cov, n_samples_per_mode)
                training_points = np.vstack((points1, points2))

            min_coord = -1.0 * distance_scale - 3 * cluster_std_dev
            max_coord = 1.0 * distance_scale + 3 * cluster_std_dev
            data_range_xy = ((min_coord, max_coord), (min_coord, max_coord))

            return {
                'training_points': training_points,
                'data_range_xy': data_range_xy
            }

    # Instantiate the dummy generator
    synthetic_generators = SyntheticGenerators()

    print("Testing Rasterizer...")
    gmm_info = synthetic_generators.get_gmm_data_for_training_and_evaluation(n_samples_per_mode=50, random_state=42, layout='square', distance_scale=1.0, cluster_std_dev=0.1)
    train_points = gmm_info['training_points']
    data_range = gmm_info['data_range_xy']
    img_size = (32, 32)

    density_tensor = rasterize_points_to_density_map(train_points, img_size, data_range, normalize_to=(-1.0, 1.0))

    print(f"Generated density map tensor shape: {density_tensor.shape}")
    print(f"Min value in map: {density_tensor.min().item():.2f}")
    print(f"Max value in map: {density_tensor.max().item():.2f}")
    assert density_tensor.shape == (1, img_size[0], img_size[1]), "Shape mismatch"
    assert density_tensor.min() >= -1.0 and density_tensor.max() <= 1.0, "Normalization range error"

    # Test with empty points
    empty_points = np.array([]).reshape(0,2)
    empty_density_tensor = rasterize_points_to_density_map(empty_points, img_size, data_range, normalize_to=(-1.0, 1.0))
    print(f"Empty points density map tensor shape: {empty_density_tensor.shape}")
    print(f"Min value in empty map: {empty_density_tensor.min().item():.2f}") # Should be -1.0 or 0 if normalize_to is [0,0]
    print(f"Max value in empty map: {empty_density_tensor.max().item():.2f}") # Should be -1.0 or 0
    assert torch.all(empty_density_tensor == -1.0) or torch.all(empty_density_tensor == 0.0), "Empty map not normalized correctly"


    print("Rasterizer test passed!")

    # Optional: Visualize the density map
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6,6))
        plt.imshow(density_tensor.squeeze().cpu().numpy(), cmap='viridis', origin='lower', extent=[data_range[0][0], data_range[0][1], data_range[1][0], data_range[1][1]])
        plt.colorbar(label='Normalized Density')
        plt.title(f'Rasterized Density Map ({img_size[0]}x{img_size[1]})')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        # plt.savefig('rasterized_gmm_density_map.png')
        print("\nPlotted rasterized GMM. If running headless, plot may not show.")
        # plt.show()
    except ImportError:
        print("\nMatplotlib not installed. Skipping density map visualization.")
    except Exception as e:
        print(f"\nError during visualization: {e}")