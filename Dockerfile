# Use an official NVIDIA PyTorch base image with CUDA 12.1 and PyTorch 2.4.0
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

LABEL maintainer="Stanley Ngugi/sngugi.research@gmail.com"
LABEL description="Docker image for Minimalist Uncertainty-Driven Explorer (MUE) project."

# Set environment variables
# For CUDA reproducibility (PyTorch 1.7+ with CUDA 10.2+)
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8
# Or :16:8. Choose one based on your needs if specific operations require it.
# The default :4096:8 is generally a good start.

# Set DEBIAN_FRONTEND to noninteractive to avoid prompts during apt-get install
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies and essential tools
RUN apt-get update && \
    apt-get install -y \
    git \
    vim \
    wget \
    build-essential \
    # Add any other system-level packages you might need (e.g., for specific dataset handling)
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
# Using --no-cache-dir to reduce image size
# Ensure your pip is up-to-date within the container
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project code into the container
COPY . .

# Expose any ports if your application needs them (e.g., for TensorBoard or a web UI)
# EXPOSE 6006 # Example for TensorBoard

# Default command to execute when the container starts (optional)
# CMD ["bash"]
# Leaving it to default to bash for interactive use is fine for development.