import wandb
import os
import random
import string

def generate_run_id(length=8):
    """Generates a short random string for run IDs."""
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

def initialize_wandb(project_name: str, run_name: str = None, config: dict = None, entity: str = None, notes: str = None, tags: list = None, job_type: str = "train"):
    """
    Initializes a Weights & Biases run.

    Args:
        project_name (str): Name of the W&B project.
        run_name (str, optional): Name of the specific run. Defaults to None (W&B auto-generates).
        config (dict, optional): Dictionary of hyperparameters to log. Defaults to None.
        entity (str, optional): W&B entity (username or team). Defaults to os.getenv("WANDB_ENTITY").
        notes (str, optional): Longer description for the run.
        tags (list, optional): List of tags for the run.
        job_type (str, optional): Type of job (e.g., 'train', 'eval', 'eda'). Defaults to "train".

    Returns:
        wandb.sdk.wandb_run.Run: The initialized W&B run object, or None if W&B is disabled.
    """
    # Check if WANDB_MODE is set to disabled
    if os.getenv("WANDB_MODE", "").lower() == "disabled":
        print("Weights & Biases logging is disabled via WANDB_MODE environment variable.")
        # Return a mock object that allows calls like .log() and .finish() without error
        class MockWandbRun:
            def __init__(self):
                self.config = config if config is not None else {}
            def log(self, data, step=None, commit=None): pass
            def finish(self): pass
            def save(self, glob_str, base_path=None, policy="live"): pass # Mock save
            def summary(self): return {} # Mock summary
            # Add other methods if your code uses them, e.g., wandb.Table, wandb.Image

        return MockWandbRun()

    if run_name is None:
        run_name = f"run_{generate_run_id()}"

    try:
        run = wandb.init(
            project=project_name,
            name=run_name,
            config=config,
            entity=entity or os.getenv("WANDB_ENTITY"), # Use provided or env var
            notes=notes,
            tags=tags,
            job_type=job_type,
            reinit=True, # Allows re-initializing in the same process (e.g., Jupyter)
            settings=wandb.Settings(start_method="thread") # Good for some environments
        )
        print(f"W&B run initialized: {run.name} (ID: {run.id}). View at: {run.url}")
        return run
    except Exception as e:
        print(f"Could not initialize W&B: {e}. Proceeding without W&B logging.")
        # Fallback to MockWandbRun if initialization fails
        class MockWandbRunOnError:
            def __init__(self):
                self.config = config if config is not None else {}
            def log(self, data, step=None, commit=None): pass
            def finish(self): pass
            def save(self, glob_str, base_path=None, policy="live"): pass
            def summary(self): return {}

        return MockWandbRunOnError()


# Example Usage (typically at the start of an experiment script)
if __name__ == '__main__':
    # To disable W&B for this test, you can uncomment the next line:
    # os.environ["WANDB_MODE"] = "disabled"

    # Ensure WANDB_API_KEY is set in your environment or you've logged in via `wandb login`
    # You might also need to set WANDB_ENTITY if it's not your default.
    # e.g., os.environ["WANDB_ENTITY"] = "your_username_or_team"

    print("Testing W&B Initialization...")
    test_config = {"learning_rate": 0.01, "epochs": 10, "dataset": "test_data"}
    
    # Test 1: Basic initialization
    print("\n--- Test 1: Basic Init ---")
    wandb_run1 = initialize_wandb(project_name="mue_test_project", run_name="test_run_1", config=test_config, tags=['test', 'basic'])
    if wandb_run1 and not isinstance(wandb_run1, (MockWandbRun, MockWandbRunOnError)): # Check if it's a real run
        wandb_run1.log({"accuracy": 0.5, "epoch": 1})
        wandb_run1.log({"accuracy": 0.7, "epoch": 2})
        wandb_run1.finish()
        print("Test 1 completed.")
    else:
        print("Test 1 used a mock W&B run or failed to init.")

    # Test 2: Auto-generated run name
    print("\n--- Test 2: Auto-generated Run Name ---")
    wandb_run2 = initialize_wandb(project_name="mue_test_project", config={"param_x": 123}, job_type="eval")
    if wandb_run2 and not isinstance(wandb_run2, (MockWandbRun, MockWandbRunOnError)):
        wandb_run2.log({"eval_metric": 99.9})
        wandb_run2.finish()
        print("Test 2 completed.")
    else:
        print("Test 2 used a mock W&B run or failed to init.")

    # Test 3: Disabled W&B
    print("\n--- Test 3: Disabled W&B (via env var) ---")
    os.environ["WANDB_MODE"] = "disabled"
    wandb_run3 = initialize_wandb(project_name="mue_test_project_disabled", config=test_config)
    wandb_run3.log({"loss": 0.1}) # Should not error
    wandb_run3.finish() # Should not error
    print("Test 3 completed (W&B was disabled, mock run used).")
    del os.environ["WANDB_MODE"] # Clean up for subsequent tests if any

    print("\nW&B initialization tests finished.")