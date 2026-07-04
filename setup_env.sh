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

# 1. Re-create the conda environment
echo "Creating Conda environment 'mdlm' with Python 3.9..."
conda deactivate || true
conda env remove -n mdlm -y || true
conda create -n mdlm python=3.9 pip -y

# 2. Set CUDA + PATH environment variables
echo "Configuring CUDA build environment..."
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$ENV_DIR/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export TORCH_CUDA_ARCH_LIST="8.6"
export MAX_JOBS=4

# 3. Install PyTorch with CUDA 12.4 wheels (pinned versions)
echo "Installing PyTorch and torchvision..."
$PIP_BIN install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# 4. Install standard pip dependencies
echo "Installing pip dependencies..."
$PIP_BIN install datasets==2.18.0 einops==0.7.0 fsspec==2024.2.0 git-lfs==1.6 h5py==3.10.0 hydra-core==1.3.2 ipdb==0.13.13 lightning==2.2.1 notebook==7.1.1 nvitop==1.3.2 omegaconf==2.3.0 packaging==23.2 pandas==2.2.1 rich==13.7.1 seaborn==0.13.2 scikit-learn==1.4.0 timm==0.9.16 transformers==4.38.2 triton==2.2.0 wandb==0.13.5

# 5. Install custom CUDA extensions using the absolute Python path
echo "Cloning and building custom CUDA extensions..."

if [ ! -d "causal-conv1d" ]; then
  git clone -b v1.1.3.post1 https://github.com/Dao-AILab/causal-conv1d.git
fi
cd causal-conv1d
rm -rf build/ dist/ *.egg-info
$PYTHON_BIN setup.py install
cd ..

if [ ! -d "mamba" ]; then
  git clone -b v1.1.4 https://github.com/state-spaces/mamba.git
fi
cd mamba
rm -rf build/ dist/ *.egg-info
$PYTHON_BIN setup.py install
cd ..

if [ ! -d "flash-attention" ]; then
  git clone -b v2.5.6 https://github.com/Dao-AILab/flash-attention.git
fi
cd flash-attention
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
