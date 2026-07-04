import os
import subprocess
import modal

# Define the Modal App
app = modal.App("mdlm-tband-training")

# Define the persistent volume for storing checkpoints and dataset caches
volume = modal.Volume.from_name("mdlm-storage", create_if_missing=True)

# Build the container image with matching PyTorch 2.4.0 and CUDA compilation tools
image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "build-essential", "clang")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        "torchaudio==2.4.0",
        index_url="https://download.pytorch.org/whl/cu121"
    )
    .pip_install("ninja", "packaging", "wheel")
    # Install FlashAttention precompiled wheel
    .pip_install("https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl")
    # Install standard python dependencies
    .pip_install(
        "datasets==2.18.0",
        "einops==0.7.0",
        "fsspec==2024.2.0",
        "git-lfs==1.6",
        "h5py==3.10.0",
        "hydra-core==1.3.2",
        "lightning==2.2.1",
        "omegaconf==2.3.0",
        "pandas==2.2.1",
        "rich==13.7.1",
        "seaborn==0.13.2",
        "scikit-learn==1.4.0",
        "timm==0.9.16",
        "transformers==4.38.2",
        "wandb==0.13.5",
        "setuptools<82"
    )
    # Build causal-conv1d locally
    .run_commands(
        "git clone -b v1.1.3.post1 https://github.com/Dao-AILab/causal-conv1d.git",
        "cd causal-conv1d && python setup.py install"
    )
    # Build mamba-ssm locally
    .run_commands(
        "git clone -b v1.1.4 https://github.com/state-spaces/mamba.git",
        "cd mamba && python setup.py install"
    )
    # Add local directory to /root/mdlm (runtime mount)
    .add_local_dir(".", remote_path="/root/mdlm")
)

# Dynamically forward WANDB_API_KEY if present in your local terminal environment
secrets = []
if "WANDB_API_KEY" in os.environ:
    secrets.append(modal.Secret.from_dict({"WANDB_API_KEY": os.environ["WANDB_API_KEY"]}))

# 1. Dataset Tokenization Function: Runs entirely on cheap Modal CPU (No GPU used)
@app.function(
    image=image,
    timeout=3600,         # 1 hour timeout
    volumes={"/root/storage": volume}
)
def tokenize():
    os.chdir("/root/mdlm")
    
    # Configure Hugging Face cache dirs to use persistent volume
    env = os.environ.copy()
    env["HF_HOME"] = "/root/storage/huggingface_home"
    env["HF_DATASETS_CACHE"] = "/root/storage/huggingface_cache"
    
    print("Starting tokenization task on Modal CPU...")
    # This runs tokenize_dataset.py which tokenizes and wraps/groups OWT
    subprocess.run(["python", "tokenize_dataset.py", "data=openwebtext-split"], env=env, check=True)
    
    volume.commit()
    print("Tokenization completed and saved to persistent volume successfully!")


# 2. Smoke Test Function: Runs entirely on cheap Modal CPU (No GPU used)
@app.function(
    image=image,
    timeout=600,          # 10 minutes timeout
    volumes={"/root/storage": volume}
)
def smoke():
    os.chdir("/root/mdlm")
    
    # Configure Hugging Face cache dirs to use persistent volume
    env = os.environ.copy()
    env["HF_HOME"] = "/root/storage/huggingface_home"
    env["HF_DATASETS_CACHE"] = "/root/storage/huggingface_cache"
    
    print("Starting offline smoke test on Modal CPU...")
    cmd = [
        "python", "-u", "-m", "main",
        "trainer.accelerator=cpu",    # Run on CPU
        "loader.batch_size=2",        # Small batch size for quick check
        "loader.eval_batch_size=2",
        "model=small",
        "data=openwebtext-split",
        "data.cache_dir=/root/storage/mdlm_data",
        "parameterization=subs",
        "model.length=1024",
        "trainer.max_steps=5",        # Only 5 steps
        "trainer.devices=1",
        "trainer.num_nodes=1",
        "training.t_band_low=0.2",
        "training.t_band_high=0.8",
        "checkpointing.save_dir=/root/storage/mdlm_checkpoints/smoke",
        "checkpointing.resume_from_ckpt=False",
        "eval.compute_generative_perplexity=False", # Skip perplexity
        "sampling.steps=10",
        "lr_scheduler.num_warmup_steps=1",
        "+wandb.offline=true"         # Run offline
    ]
    
    subprocess.run(cmd, env=env, check=True)
    volume.commit()
    print("Smoke test on CPU passed successfully!")


# 3. Full Training Function: Runs on cost-effective L4 GPU ($0.72/hour)
@app.function(
    image=image,
    gpu="L4",             # L4 is extremely cost-effective ($0.72/hr) and perfect for this job
    timeout=7200,         # 2 hours timeout max (safeguards your budget)
    volumes={"/root/storage": volume},
    secrets=secrets
)
def train():
    os.chdir("/root/mdlm")

    # Define directories inside the persistent volume
    cache_dir = "/root/storage/mdlm_data"
    save_dir = "/root/storage/mdlm_checkpoints/tband_0.2_0.8"
    
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # Set HF cache environment variables to the persistent volume
    env = os.environ.copy()
    env["HF_HOME"] = "/root/storage/huggingface_home"
    env["HF_DATASETS_CACHE"] = "/root/storage/huggingface_cache"

    cmd = [
        "python", "-u", "-m", "main",
        "trainer.accelerator=gpu",
        "loader.batch_size=16",
        "loader.eval_batch_size=16",
        "model=small",
        "data=openwebtext-split",
        f"data.cache_dir={cache_dir}",
        "wandb.project=sm-mdlm",
        "wandb.entity=slerp-on-smdlm",
        "wandb.name=mdlm-owt-tband-0.2-0.8",
        "parameterization=subs",
        "model.length=1024",
        "trainer.max_steps=2000",      # Full 2k steps run
        "trainer.devices=1",
        "trainer.num_nodes=1",
        "training.t_band_low=0.2",
        "training.t_band_high=0.8",
        f"checkpointing.save_dir={save_dir}",
        "checkpointing.resume_from_ckpt=False",
        "eval.compute_generative_perplexity=True",
        "sampling.steps=1000",
        "lr_scheduler.num_warmup_steps=200"
    ]

    # If no W&B secret is forwarded, run offline to avoid blocking
    if not secrets:
        cmd.append("+wandb.offline=true")
        print("No WANDB_API_KEY found in local environment. Running training offline...")

    print("Starting full GPU training command on Modal L4...")
    subprocess.run(cmd, env=env, check=True)
    
    # Commit volume changes explicitly
    volume.commit()
    print("Training complete! Checkpoints successfully saved to persistent volume.")
