#!/bin/bash

#SBATCH --ntasks=1

#SBATCH -A uvailp
#SBATCH -p gpu
#SBATCH --gres=gpu:8  # Specify the 80GB version of the A100
#SBATCH --cpus-per-task=2
#SBATCH --time=2-00:00:00
#SBATCH --chdir=/scratch/jax4zk/intrinsic-dimension-of-learning
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err


module load anaconda
module load cuda
module load cudnn

conda activate lid
python run_few_ft_exp.py yaml/few_sample_ft_exp.yaml