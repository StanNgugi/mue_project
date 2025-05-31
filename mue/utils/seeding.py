import random
import numpy as np
import torch
import os

def set_global_seeds(seed_value: int):
    """
    Set seeds for all relevant random number generators to ensure reproducibility.

    Args:
        seed_value (int): The integer value to use as the seed.
    """
    if seed_value is None:
        # Generate a random seed if None is provided, useful for hyperparameter search
        # where each run should be different but still logged.
        seed_value = random.randint(0, 2**32 - 1)
        print(f"No seed provided, using randomly generated seed: {seed_value}")
    else:
        print(f"Using global seed: {seed_value}")

    os.environ['PYTHONHASHSEED'] = str(seed_value)  # For Python's hash-based operations
    random.seed(seed_value)  # Python's built-in random module
    np.random.seed(seed_value)  # NumPy
    torch.manual_seed(seed_value)  # PyTorch CPU

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)  # PyTorch current GPU
        torch.cuda.manual_seed_all(seed_value)  # PyTorch all GPUs (if multi-GPU)

    # Note: For full CUDA reproducibility, also ensure deterministic algorithms are used:
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True) (PyTorch 1.8+)
    # And set CUBLAS_WORKSPACE_CONFIG environment variable (done in Dockerfile).
    # These are typically set at the beginning of your main experiment script.

if __name__ == '__main__':
    print("Testing global seed setting...")
    initial_random = random.random()
    initial_np_random = np.random.rand()
    initial_torch_random = torch.rand(1)

    set_global_seeds(42)
    seeded_random1 = random.random()
    seeded_np_random1 = np.random.rand()
    seeded_torch_random1 = torch.rand(1)

    # Re-seed to check consistency
    set_global_seeds(42)
    seeded_random2 = random.random()
    seeded_np_random2 = np.random.rand()
    seeded_torch_random2 = torch.rand(1)

    # Seed with a different value
    set_global_seeds(101)
    diff_seeded_random = random.random()
    diff_seeded_np_random = np.random.rand()
    diff_seeded_torch_random = torch.rand(1)

    print(f"Initial Python random: {initial_random}")
    print(f"Seeded (42) Python random 1: {seeded_random1}")
    print(f"Seeded (42) Python random 2 (should be same as 1): {seeded_random2}")
    assert seeded_random1 == seeded_random2, "Python random seeding failed"
    print(f"Seeded (101) Python random (should be different): {diff_seeded_random}")
    assert seeded_random1 != diff_seeded_random, "Python random seeding failed (diff seed)"

    print(f"\nInitial NumPy random: {initial_np_random}")
    print(f"Seeded (42) NumPy random 1: {seeded_np_random1}")
    print(f"Seeded (42) NumPy random 2 (should be same as 1): {seeded_np_random2}")
    assert seeded_np_random1 == seeded_np_random2, "NumPy random seeding failed"
    print(f"Seeded (101) NumPy random (should be different): {diff_seeded_np_random}")
    assert seeded_np_random1 != diff_seeded_np_random, "NumPy random seeding failed (diff seed)"

    print(f"\nInitial PyTorch random: {initial_torch_random.item()}")
    print(f"Seeded (42) PyTorch random 1: {seeded_torch_random1.item()}")
    print(f"Seeded (42) PyTorch random 2 (should be same as 1): {seeded_torch_random2.item()}")
    assert seeded_torch_random1.item() == seeded_torch_random2.item(), "PyTorch random seeding failed"
    print(f"Seeded (101) PyTorch random (should be different): {diff_seeded_torch_random.item()}")
    assert seeded_torch_random1.item() != diff_seeded_torch_random.item(), "PyTorch random seeding failed (diff seed)"

    print("\nSeed setting test passed!")