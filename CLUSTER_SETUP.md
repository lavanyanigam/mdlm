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

* **Current Active Cache Path**: `/tmp/mdlm_data`
* **Hugging Face Cache**: `/tmp/huggingface_cache` (redirected via `HF_DATASETS_CACHE` env var)
* **Why `/tmp`?** Using `/tmp` bypasses your personal 320 GB home directory quota. It resides on the high-speed local NVMe storage of the compute node (with over 270 GB of free space), ensuring optimal performance.
* **Fast Tokenization**: Tokenization leverages multi-processing (`num_proc=num_workers`) and PyArrow's zero-copy memory mapping on the local SSD. Disk caching is kept enabled (`load_from_cache_file=True`) so that if a training job is restarted or resumed on the same node, it loads the tokenized dataset **instantly** without re-processing.

### How to Clean Up Half-Tokenized / Aborted Datasets
If a tokenization run gets aborted mid-way, Hugging Face `datasets` can leave behind lock files or half-written temporary cache files that consume space. To clear them and start clean, run:

```bash
# Clear all Hugging Face temporary files and cache on the node
rm -rf /tmp/huggingface_cache/*
rm -rf /tmp/huggingface_home/*

# Clear the tokenized dataset cache on the node
rm -rf /tmp/mdlm_data/*
```

### Cleaning Up Old/Wasted Checkpoints
Model checkpoints (`.ckpt` files) consume several gigabytes each. Since they are saved to `/tmp/mdlm_checkpoints` to avoid quota limits, you should clean up any old, unneeded runs:

```bash
# Remove all checkpoints in the /tmp checkpoints folder
rm -rf /tmp/mdlm_checkpoints/*
```

> [!IMPORTANT]
> Because `/tmp` is temporary and local to the node, make sure to copy your final model checkpoint (e.g. `last.ckpt`) back to your home folder `~/mdlm/` once training completes:
> ```bash
> cp /tmp/mdlm_checkpoints/<run_name>/checkpoints/last.ckpt ~/mdlm/
> ```

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
