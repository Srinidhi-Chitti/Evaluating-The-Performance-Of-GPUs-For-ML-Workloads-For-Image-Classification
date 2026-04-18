# GPU Performance Evaluation for Image Classification Workloads

## Overview
This project evaluates GPU performance for machine learning workloads in image classification. It benchmarks and profiles Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) using JAX and Flax, focusing on execution time, accuracy, and resource utilization. The project also applies optimization techniques such as JIT compilation and kernel fusion to improve efficiency.

---

## Problem Statement
Current machine learning workflows lack a systematic approach to evaluating GPU performance. This leads to:
- Limited visibility into training time and resource usage
- Poor understanding of performance bottlenecks
- Inefficient utilization of GPU capabilities

---

## Objectives
- Evaluate CNN and ViT models on GPU
- Benchmark performance across datasets
- Identify system and model-level bottlenecks
- Optimize training pipelines and measure improvements

---

## System Design
The project is divided into three components:

### 1. Benchmarking
Measures:
- Training time
- Accuracy
- Resource utilization

### 2. Profiling
Identifies:
- GPU/CPU bottlenecks
- Expensive operations

### 3. Optimization
Applies:
- JIT compilation
- Kernel fusion
- Memory transfer reduction

---

## Tech Stack
- JAX
- Flax
- CUDA (NVIDIA GPU)
- XProf
- WSL (Windows Subsystem for Linux)

---

## Environment Setup

### Prerequisites
- Python 3.12
- NVIDIA GPU with CUDA 12 support
- WSL (recommended for Windows users)

### Setup Steps

```bash
mkdir -p ~/Projects/evaluating-gpus
cd ~/Projects/evaluating-gpus

python3.12 -m venv jax-env
source jax-env/bin/activate

pip install --upgrade pip
pip install "jax[cuda12]"
pip install flax

### Verify Installation
pip show jax
nvidia-smi
nvcc --version

### Common Commands
# Activate environment
source jax-env/bin/activate

# Create directories
mkdir -p src/<folder_name>

Datasets and Models
Datasets
MNIST
CIFAR-10
Models
Convolutional Neural Network (CNN)
Vision Transformer (ViT)

Benchmarking Methodology:
The benchmarking process follows best practices from JAX:
-Use jax.jit for compilation
-Use .block_until_ready() to handle asynchronous execution
-Use appropriate data types (float32 vs float64)
-Minimize CPU-to-GPU data transfer

Profiling:
Both manual and programmatic profiling techniques are supported. This project primarily uses manual profiling with XProf.
-Start profiler server in code:
import jax.profiler
jax.profiler.start_server(9999)
-Run the training script:
python -m <filepath>
-Collect profiling data:
python -m jax.collect_profile 9999 5000 --log_dir=/tmp/profile-data --no_perfetto_link
-Launch XProf:
xprof --port 8791 /tmp/profile-data

Optimization Strategy
Standard Pipeline:
Each step is executed independently:
Normalization
Blurring
Edge detection
Model inference

This results in:
Multiple kernel launches
Repeated memory transfers
X1 = X / 255
X2 = X1 * K_blur
X3 = X2 * K_sobel
Y  = f(X3)

Total execution time:T_total = Σ (T_c(i) + T_m(i))

Optimized Pipeline (JIT Fusion)
All operations are fused into a single computation graph using JIT compilation. T_total = T_c(fused) + T_m(minimal)

Key Improvements
Reduced kernel launches
Lower memory overhead
Improved execution speed
