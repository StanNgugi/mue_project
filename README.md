# Minimalist Uncertainty-Driven Explorer (MUE) for Diffusion Models

This project implements the Minimalist Uncertainty-Driven Explorer (MUE) framework, designed to endow pre-trained diffusion models with the capability for autonomous exploration of novel, coherent regions of their learned data manifold. MUE operates without requiring architectural modifications to the base model, without model retraining, and without reliance on complex external supervision signals or auxiliary networks.

The core hypothesis is that local instabilities in the denoising process, quantified by a Denoising Instability Score (DIS), can serve as a proxy for epistemic-like uncertainty. This DIS signal is then used by Instability-Biased Guidance (IBG) to steer the generative process towards these uncertain, potentially novel, regions.

This repository contains the code for:
1.  Training simple DDPMs on synthetic 2D datasets (e.g., Gaussian Mixture Models with missing modes).
2.  Calculating the Denoising Instability Score (DIS) for various models and datasets.
3.  Implementing Instability-Biased Guidance (IBG) for exploration.
4.  Evaluating MUE against baselines on synthetic and real-world image datasets (e.g., CIFAR-10).
5.  Reproducing the experiments outlined in the MUE research framework.

## Project Structure

(A brief overview of the main directories will be more fleshed out as we build them, but for now:)
- `configs/`: YAML configuration files for experiments.
- `data/`: Scripts related to data generation, processing, and PyTorch Datasets.
- `experiments/`: Main executable scripts for running training, validation, and evaluation.
- `mue/`: Core MUE library code (DIS, IBG, pipelines, metrics, utilities).
- `results/`: Directory for storing outputs like generated images, plots, and metric scores (should be in `.gitignore` for large files).
- `Dockerfile`: Defines the Docker container for a reproducible environment.
- `requirements.txt`: Python package dependencies.

## Setup and Installation

This project uses Docker to ensure a reproducible environment, especially for managing PyTorch and CUDA versions.

**Prerequisites:**
- Docker installed on your system.
- NVIDIA GPU with appropriate drivers installed (for GPU acceleration). (See "NVIDIA Driver Compatibility" below).
- Git (for cloning the repository).
- Access to a terminal/shell (e.g., Ubuntu terminal).

**NVIDIA Driver Compatibility:**
The chosen Docker image will bundle a specific CUDA toolkit version (e.g., CUDA 12.1). Your host machine's NVIDIA driver must be compatible with this CUDA version.
- For CUDA 12.1, a common minimum Linux driver version is >=530.30.02.
- You can check your driver version by running `nvidia-smi` in your host terminal.
- If running on a cloud platform like RunPod, ensure the selected GPU instance has a compatible driver.

**Steps:**

1.  **Clone the Repository:**
    ```bash
    git clone <your-github-repository-url>
    cd <your-repository-name>
    ```

2.  **Build the Docker Image:**
    (The `Dockerfile` will be provided in the next chunk. This command assumes it's in the project root.)
    ```bash
    docker build -t mue_project .
    ```

3.  **Run the Docker Container:**
    This command starts an interactive session within the Docker container, mounts the current project directory into `/app` inside the container, enables GPU access, and sets up an example environment variable for Weights & Biases (if used).

    ```bash
    docker run --gpus all -it --rm \
        -v "$(pwd)":/app \
        -w /app \
        -e WANDB_API_KEY='YOUR_WANDB_API_KEY_IF_APPLICABLE' \
        mue_project bash
    ```
    - `--rm`: Automatically removes the container when it exits.
    - `-w /app`: Sets the working directory inside the container to `/app`.
    - You can add other port mappings (`-p`) or environment variables (`-e`) as needed.

4.  **Inside the Docker Container:**
    All subsequent commands for installing Python packages (if any beyond the Dockerfile) or running experiments should be executed inside this Docker container's shell. The `requirements.txt` file (to be provided) will list Python dependencies. These are typically installed when the Docker image is built.

## Running Experiments

Experiments are generally run via Python scripts located in the `experiments/` directory, using configuration files from `configs/`.

Example (actual script and config names will vary):
```bash
# Inside the Docker container
python experiments/run_phase1_gmm_ddpm_train.py --config configs/phase1_gmm_ddpm_train_config.yaml