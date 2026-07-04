import os
import subprocess
import modal

# Define the Modal App
app = modal.App("mdlm-tband-training")

# Define the persistent volume for storing checkpoints and dataset caches
volume = modal.Volume.from_name("mdlm-storage", create_if_missing=True)

# Build the container image with matching PyTorch 2.4.0 and CUDA compilation tools
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "build-essential")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        "torchaudio==2.4.0",
        index_url="https://download.pytorch.org/whl/cu121"
    )
    .pip_install("ninja", "packaging")
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
)

# We mount your local repository directory into /root/mdlm inside the container.
# We also mount the persistent volume to /root/storage for datasets and checkpoints.
@app.function(
    image=image,
    gpu="A10G",          # A10G is $1.10/hour (very cheap and plenty of power for 2k steps)
    timeout=7200,        # 2 hours timeout max (saves budget in case of hang)
    mounts=[modal.Mount.from_local_dir(".", remote_path="/root/mdlm")],
    volumes={"/root/storage": volume},
    secrets=[modal.Secret.from_name("my-wandb-secret")] # Add wandb secret if logging to wandb
)
def train(smoke: bool = False):
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

    # Default parameters for a full training run
    max_steps = 2000
    warmup_steps = 200
    compute_perplexity = "True"
    sampling_steps = 1000
    wandb_project = "sm-mdlm"
    wandb_name = "mdlm-owt-tband-0.2-0.8"

    # If running a smoke test, override with extremely lightweight settings
    if smoke:
        print("====== RUNNING LIGHTWEIGHT SMOKE TEST ======")
        max_steps = 5
        warmup_steps = 1
        compute_perplexity = "False"
        sampling_steps = 10
        wandb_project = "sm-mdlm-smoke"
        wandb_name = "mdlm-smoke-test"

    cmd = [
        "python", "-u", "-m", "main",
        "loader.batch_size=16",
        "loader.eval_batch_size=16",
        "model=small",
        "data=openwebtext-split",
        f"data.cache_dir={cache_dir}",
        f"wandb.project={wandb_project}",
        "wandb.entity=slerp-on-smdlm",
        f"wandb.name={wandb_name}",
        "parameterization=subs",
        "model.length=1024",
        f"trainer.max_steps={max_steps}",
        "trainer.devices=1",
        "trainer.num_nodes=1",
        "training.t_band_low=0.2",
        "training.t_band_high=0.8",
        f"checkpointing.save_dir={save_dir}",
        "checkpointing.resume_from_ckpt=False",
        f"eval.compute_generative_perplexity={compute_perplexity}",
        f"sampling.steps={sampling_steps}",
        f"lr_scheduler.num_warmup_steps={warmup_steps}"
    ]

    if smoke:
        cmd.append("+wandb.offline=true")  # Run offline for smoke test to avoid login blocks

    print(f"Starting training command on Modal (smoke={smoke})...")
    subprocess.run(cmd, env=env, check=True)
    
    # Commit volume changes explicitly
    volume.commit()
    print("Training finished successfully! Persistent volume committed.")
