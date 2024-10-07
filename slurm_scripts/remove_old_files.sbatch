#!/bin/bash

#SBATCH --ntasks=1

#SBATCH -A uvailp
#SBATCH -p standard
#SBATCH --cpus-per-task=2
#SBATCH --time=3-00:00:00
#SBATCH --chdir=/scratch/jax4zk/
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err


# rm -rf icl-lid/

rsync -a --delete empty_dir/ icl-lid/