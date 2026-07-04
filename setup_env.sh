#!/bin/bash
# Exit immediately if any command fails
set -e

echo "===================================================="
echo "Starting automated environment setup for MDLM-tband"
echo "===================================================="

# 1. Initialize Conda for the script session
CONDA_BASE_PATH=$(conda info --base)
source "$CONDA_BASE_PATH/etc/profile.d/conda.sh"

# 2. Re-create the conda environment
echo "Creating Conda environment 'mdlm' with Python 3.9..."
conda deactivate || true
conda env remove -n mdlm -y || true
conda create -n mdlm python=3.9 pip -y
conda activate mdlm

# 3. Install PyTorch, Torchvision, and CUDA components via Conda
echo "Installing PyTorch, torchvision, and CUDA dependencies..."
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 cuda-nvcc=12.4 -c pytorch -c nvidia -y

# 4. Set Environment Variables for Fast CUDA Extension Compilation
echo "Configuring CUDA build environment..."
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Target RTX A6000 architecture (sm_86) to speed up compile time by 20x
export TORCH_CUDA_ARCH_LIST="8.6"
export MAX_JOBS=4

# 5. Install Standard Pip Dependencies
echo "Installing pip dependencies..."
pip install datasets==2.18.0 einops==0.7.0 fsspec==2024.2.0 git-lfs==1.6 h5py==3.10.0 hydra-core==1.3.2 ipdb==0.13.13 lightning==2.2.1 notebook==7.1.1 nvitop==1.3.2 omegaconf==2.3.0 packaging==23.2 pandas==2.2.1 rich==13.7.1 seaborn==0.13.2 scikit-learn==1.4.0 timm==0.9.16 transformers==4.38.2 triton==2.2.0 wandb==0.13.5

# 6. Install Custom CUDA Extensions from Clean GitHub checkouts
echo "Cloning and building custom CUDA extensions..."

# Causal Conv1D
if [ ! -d "causal-conv1d" ]; then
  git clone -b v1.1.3.post1 https://github.com/Dao-AILab/causal-conv1d.git
fi
cd causal-conv1d
rm -rf build/ dist/ *.egg-info
python setup.py install
cd ..

# Mamba SSM
if [ ! -d "mamba" ]; then
  git clone -b v1.1.4 https://github.com/state-spaces/mamba.git
fi
cd mamba
rm -rf build/ dist/ *.egg-info
python setup.py install
cd ..

# Flash Attention
if [ ! -d "flash-attention" ]; then
  git clone -b v2.5.6 https://github.com/Dao-AILab/flash-attention.git
fi
cd flash-attention
rm -rf build/ dist/ *.egg-info
python setup.py install
cd ..

echo "===================================================="
echo "Setup complete! Environment 'mdlm' is ready."
echo "===================================================="
