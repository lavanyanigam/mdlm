# Slurm Cluster Environment Setup & Training Guide

This guide documents how to set up the environment, manage disk quotas, clean up checkpoints, and run training on the GPU cluster.

---

## 1. Environment Installation
To install the environment without hitting Out-of-Memory (OOM) or CPU throttling limits on the login node, compile the custom packages inside a Slurm interactive compute node.

1. **Pull the latest code updates**:
   ```bash
   git pull origin master
   ```

2. **Launch an interactive compute node session** (runs on a GPU node with sufficient RAM):
   ```bash
   srun -N 1 --ntasks-per-node=1 --cpus-per-task=8 --gres=gpu:1 --mem=32G -t 00:15:00 --pty bash
   ```

3. **Run the automated setup script** (completes in ~2 minutes because it leverages a precompiled FlashAttention wheel for PyTorch 2.4.0 / Python 3.10):
   ```bash
   bash setup_env.sh
   ```

---

## 2. Dataset Caching & Disk Quota Management
The OpenWebText dataset is large and can easily exceed user storage quotas in your home directory `/home/aryan_s2/`.

* **Current Active Cache Path**: `/home/aryan_s2/mdlm_data`
* **Why?** Storing in your home folder ensures the tokenized cache is permanent and won't be deleted on system reboots.
* **Configuration**: This is pre-configured in the launch script via the `data.cache_dir=/home/aryan_s2/mdlm_data` override.

### Cleaning Up Old/Wasted Checkpoints
Model checkpoints (`.ckpt` files) consume several gigabytes each. To free up your home directory quota, run the following commands:

* **Remove all checkpoints in the temporary checkpoints folder**:
   ```bash
   rm -rf /home/aryan_s2/checkpoints/*.ckpt
   ```

* **Remove old checkpoints inside Hydra output folders**:
   ```bash
   rm -rf /home/aryan_s2/mdlm/outputs/*/checkpoints/*.ckpt
   ```

---

## 3. Submitting the Training Job
The training script `scripts/train_owt_tband.sh` is configured to run on **1 GPU** for **2000 steps** with a correct warmup profile of **200 steps** (10% of total steps), logging to your target W&B space.

To run it:
```bash
sbatch scripts/train_owt_tband.sh
```

To watch the live logs:
```bash
tail -f watch_folder/train_mdlm_tband_<JOBID>.out
```

---

## 4. Weights & Biases Logging
Logs and plots are streamed live to:
* **W&B Team/Entity**: `slerp-on-smdlm`
* **W&B Project**: `sm-mdlm`
* **W&B Run Name**: `mdlm-owt-tband-0.2-0.8`

Live Link: [https://wandb.ai/slerp-on-smdlm/sm-mdlm](https://wandb.ai/slerp-on-smdlm/sm-mdlm)
