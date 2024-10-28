#!/bin/bash

#SBATCH --ntasks=1

#SBATCH -A uvailp
#SBATCH -p gpu
#SBATCH --gres=gpu:a6000:5   # Specify the 80GB version of the A100
#SBATCH --cpus-per-task=2
#SBATCH --time=1-00:00:00
#SBATCH --chdir=/scratch/jax4zk/intrinsic-dimension-of-learning
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err


module load cuda
module load cudnn
module load miniforge


conda activate lid

python detailed_finetune.py yaml/detailed_ft_1.yaml