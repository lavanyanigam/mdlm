#!/bin/bash
# Exit immediately if any command fails
set -e
echo "===================================================="
echo "Starting automated environment setup for MDLM-tband"
echo "===================================================="

# Define absolute paths to prevent conda activate shell issues
ENV_DIR="/home/aryan_s2/.conda/envs/mdlm"
PYTHON_BIN="$ENV_DIR/bin/python"
PIP_BIN="$ENV_DIR/bin/pip"

# 1. Re-create the conda environment with Python 3.10 to match official wheels
echo "Creating Conda environment 'mdlm' with Python 3.10..."
conda deactivate || true
conda env remove -n mdlm -y || true
conda create -n mdlm python=3.10 pip -y

# 2. Set CUDA + PATH environment variables
echo "Configuring CUDA build environment..."
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$ENV_DIR/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export TORCH_CUDA_ARCH_LIST="8.6"
export MAX_JOBS=2

# 3. Install PyTorch 2.4.0 with CUDA 12.1 wheels (fully compatible with 12.8 driver)
echo "Installing PyTorch 2.4.0 and torchvision..."
$PIP_BIN install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install ninja and packaging for fast parallel compiling of local modules
$PIP_BIN install ninja packaging

# 4. Install standard pip dependencies
echo "Installing pip dependencies..."
$PIP_BIN install datasets==2.18.0 einops==0.7.0 fsspec==2024.2.0 git-lfs==1.6 h5py==3.10.0 hydra-core==1.3.2 ipdb==0.13.13 lightning==2.2.1 notebook==7.1.1 nvitop==1.3.2 omegaconf==2.3.0 pandas==2.2.1 rich==13.7.1 seaborn==0.13.2 scikit-learn==1.4.0 timm==0.9.16 transformers==4.38.2 wandb==0.13.5

# 5. Install custom CUDA extensions
echo "Installing custom CUDA extensions..."

# Install Flash Attention from precompiled GitHub Release wheel (installs in seconds!)
echo "Installing Flash Attention via precompiled wheel..."
$PIP_BIN install https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Causal Conv1D (builds locally - very fast, <1 minute)
if [ ! -d "causal-conv1d" ]; then
  git clone -b v1.1.3.post1 https://github.com/Dao-AILab/causal-conv1d.git
fi
cd causal-conv1d
rm -rf build/ dist/ *.egg-info
$PYTHON_BIN setup.py install
cd ..

# Mamba SSM (builds locally - very fast, <1 minute)
if [ ! -d "mamba" ]; then
  git clone -b v1.1.4 https://github.com/state-spaces/mamba.git
fi
cd mamba
rm -rf build/ dist/ *.egg-info
$PYTHON_BIN setup.py install
cd ..

# 6. Verify everything imports cleanly before declaring victory
echo "Verifying installation..."
$PYTHON_BIN -c "
import torch, causal_conv1d, mamba_ssm, flash_attn
print('torch:', torch.__version__, '| CUDA:', torch.version.cuda)
print('CXX11 ABI:', torch._C._GLIBCXX_USE_CXX11_ABI)
print('All extensions imported successfully.')
"

echo "===================================================="
echo "Setup complete! Environment 'mdlm' is ready."
echo "===================================================="
