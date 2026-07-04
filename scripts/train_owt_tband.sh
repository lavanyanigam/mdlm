#!/bin/bash
#SBATCH -J train_mdlm_tband           # Job name
#SBATCH -o watch_folder/%x_%j.out     # output file (%j expands to jobID)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --mem=64G                     # server memory requested
#SBATCH -t 04:00:00                   # Time limit (hh:mm:ss)
#SBATCH --partition=gpu               # Request partition
#SBATCH --ntasks-per-node=2           # 2 tasks per node (one per GPU)
#SBATCH --gres=gpu:2                  # Type/number of GPUs needed
#SBATCH --cpus-per-task=4             # CPUs per task
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon pre-emption
 
# Explicitly set the environment instead of inheriting the login shell
# (--get-user-env removed on purpose: it can silently reintroduce a
# system Python / mismatched CUDA libs on the compute node)
export CUDA_HOME=/usr/local/cuda-12.8
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export PATH="/home/aryan_s2/.conda/envs/mdlm/bin:$PATH"
 
srun /home/aryan_s2/.conda/envs/mdlm/bin/python -u -m main \
loader.batch_size=16 \
loader.eval_batch_size=16 \
model=small \
data=openwebtext-split \
wandb.name=mdlm-owt-tband-0.2-0.8 \
parameterization=subs \
model.length=1024 \
trainer.max_steps=2000 \
trainer.devices=2 \
trainer.num_nodes=1 \
training.t_band_low=0.2 \
training.t_band_high=0.8 \
checkpointing.save_dir=/home/aryan_s2/mdlm_checkpoints/tband_0.2_0.8 \
eval.compute_generative_perplexity=True \
sampling.steps=1000
