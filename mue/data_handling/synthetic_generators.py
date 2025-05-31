import numpy as np
from sklearn.datasets import make_blobs
from typing import Tuple, List, Optional, Dict, Any

def generate_gmm_points(
    n_samples_per_mode: List[int],
    centers: List[Tuple[float, float]],
    cluster_std: List[float],
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates points from a Gaussian Mixture Model using sklearn.make_blobs.

    Args:
        n_samples_per_mode (List[int]): Number of samples for each mode.
        centers (List[Tuple[float, float]]): List of (x, y) coordinates for the center of each mode.
        cluster_std (List[float]): Standard deviation for each mode.
        random_state (Optional[int]): Seed for reproducibility.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - points (np.ndarray): Array of shape (total_samples, 2) containing (x, y) coordinates.
            - labels (np.ndarray): Array of shape (total_samples,) containing the mode index for each point.
    """
    if not (len(n_samples_per_mode) == len(centers) == len(cluster_std)):
        raise ValueError("n_samples_per_mode, centers, and cluster_std must have the same length.")

    points, labels = make_blobs(
        n_samples=n_samples_per_mode,
        centers=np.array(centers),
        cluster_std=cluster_std,
        random_state=random_state,
        n_features=2
    )
    return points, labels

def get_gmm_data_for_training_and_evaluation(
    n_total_modes: int = 4,
    n_samples_per_mode: int = 2000,
    missing_mode_idx: int = 0,
    layout: str = 'square', # 'square', 'circle', or provide custom centers
    distance_scale: float = 3.0,
    cluster_std_dev: float = 0.3,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Generates GMM data, separates training data by excluding one mode, and provides info
    about all modes for consistent rasterization and evaluation.

    Args:
        n_total_modes (int): Total number of modes to generate initially.
        n_samples_per_mode (int): Number of samples to generate for each mode.
        missing_mode_idx (int): Index of the mode to exclude from the training set.
                                 If None, all modes are included in training data.
        layout (str): Predefined layout for mode centers ('square', 'circle').
        distance_scale (float): Scaling factor for mode center positions.
        cluster_std_dev (float): Standard deviation for all clusters.
        random_state (Optional[int]): Seed for reproducibility.

    Returns:
        Dict[str, Any]: A dictionary containing:
            'all_points': np.ndarray of all generated points (N_total, 2).
            'all_labels': np.ndarray of labels for all_points (N_total,).
            'training_points': np.ndarray of points for training (excluding missing mode).
            'training_labels': np.ndarray of labels for training_points.
            'missing_mode_points': np.ndarray of points belonging to the missing mode.
            'missing_mode_center': Tuple[float,float] center of the missing mode.
            'mode_centers': List of all mode centers.
            'data_range_xy': Tuple[Tuple[float,float], Tuple[float,float]] ((min_x, max_x), (min_y, max_y))
                             calculated from ALL generated points.
            'config': Dictionary of generation parameters.
    """
    if missing_mode_idx is not None and not (0 <= missing_mode_idx < n_total_modes):
        raise ValueError(f"missing_mode_idx must be between 0 and {n_total_modes - 1}, or None.")

    # Define mode centers based on layout
    centers = []
    if layout == 'square':
        if n_total_modes != 4:
            print(f"Warning: Square layout typically uses 4 modes. Using first {n_total_modes} of a square pattern.")
        possible_centers = [
            (-distance_scale, -distance_scale), (distance_scale, -distance_scale),
            (-distance_scale, distance_scale), (distance_scale, distance_scale)
        ]
        centers = possible_centers[:n_total_modes]
    elif layout == 'circle':
        for i in range(n_total_modes):
            angle = 2 * np.pi * i / n_total_modes
            centers.append((distance_scale * np.cos(angle), distance_scale * np.sin(angle)))
    else:
        raise ValueError(f"Unknown layout: {layout}. Choose 'square', 'circle', or implement custom centers.")
    
    if len(centers) < n_total_modes:
         raise ValueError(f"Layout '{layout}' did not generate enough centers ({len(centers)}) for {n_total_modes} modes.")


    all_points, all_labels = generate_gmm_points(
        n_samples_per_mode=[n_samples_per_mode] * n_total_modes,
        centers=centers,
        cluster_std=[cluster_std_dev] * n_total_modes,
        random_state=random_state
    )

    # Determine data range from ALL points for consistent rasterization
    min_x, max_x = all_points[:, 0].min(), all_points[:, 0].max()
    min_y, max_y = all_points[:, 1].min(), all_points[:, 1].max()
    padding_x = (max_x - min_x) * 0.1 # Add 10% padding
    padding_y = (max_y - min_y) * 0.1
    data_range_xy = ((min_x - padding_x, max_x + padding_x), (min_y - padding_y, max_y + padding_y))


    if missing_mode_idx is not None:
        training_mask = all_labels != missing_mode_idx
        training_points = all_points[training_mask]
        training_labels = all_labels[training_mask]
        
        missing_mode_mask = all_labels == missing_mode_idx
        missing_mode_points = all_points[missing_mode_mask]
        missing_mode_center = centers[missing_mode_idx]
    else:
        training_points = all_points
        training_labels = all_labels
        missing_mode_points = np.array([]).reshape(0,2)
        missing_mode_center = None


    return {
        'all_points': all_points,
        'all_labels': all_labels,
        'training_points': training_points,
        'training_labels': training_labels,
        'missing_mode_points': missing_mode_points,
        'missing_mode_center': missing_mode_center,
        'mode_centers': centers,
        'data_range_xy': data_range_xy,
        'config': {
            'n_total_modes': n_total_modes,
            'n_samples_per_mode': n_samples_per_mode,
            'missing_mode_idx': missing_mode_idx,
            'layout': layout,
            'distance_scale': distance_scale,
            'cluster_std_dev': cluster_std_dev,
            'random_state': random_state
        }
    }

if __name__ == '__main__':
    gmm_data_dict = get_gmm_data_for_training_and_evaluation(
        n_total_modes=4,
        n_samples_per_mode=100, # Small for quick test
        missing_mode_idx=0,
        layout='square',
        distance_scale=2.0,
        cluster_std_dev=0.2,
        random_state=42
    )

    print(f"Generated data for MUE GMM experiment:")
    print(f"  Total points generated: {len(gmm_data_dict['all_points'])}")
    print(f"  Training points: {len(gmm_data_dict['training_points'])}")
    print(f"  Missing mode ({gmm_data_dict['config']['missing_mode_idx']}) points: {len(gmm_data_dict['missing_mode_points'])}")
    print(f"  Missing mode center: {gmm_data_dict['missing_mode_center']}")
    print(f"  All mode centers: {gmm_data_dict['mode_centers']}")
    print(f"  Data range (x, y): {gmm_data_dict['data_range_xy']}")

    # Example visualization (requires matplotlib)
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.scatter(gmm_data_dict['all_points'][:, 0], gmm_data_dict['all_points'][:, 1], c=gmm_data_dict['all_labels'], s=5, cmap='viridis')
        plt.title('All Generated GMM Points')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        if len(gmm_data_dict['training_points']) > 0 :
             plt.scatter(gmm_data_dict['training_points'][:, 0], gmm_data_dict['training_points'][:, 1], c=gmm_data_dict['training_labels'][gmm_data_dict['training_labels'] != gmm_data_dict['config']['missing_mode_idx']], s=5, cmap='viridis') # Remap labels if needed for consistent coloring
        if len(gmm_data_dict['missing_mode_points']) > 0:
            plt.scatter(gmm_data_dict['missing_mode_points'][:, 0], gmm_data_dict['missing_mode_points'][:, 1], s=5, color='red', label=f'Missing Mode {gmm_data_dict["config"]["missing_mode_idx"]}')
            plt.legend()
        plt.title('Training Data (Missing Mode in Red if Plotted)')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        
        plt.tight_layout()
        # plt.savefig('gmm_data_visualization.png') # Optional: save the plot
        print("\nPlotted GMM data. If running in a headless environment, plot may not show but no error means it worked.")
        # plt.show() # Uncomment if running in an environment that supports GUI plots
    except ImportError:
        print("\nMatplotlib not installed. Skipping GMM data visualization.")
    except Exception as e:
        print(f"\nError during visualization: {e}")