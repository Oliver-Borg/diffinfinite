#!/bin/bash
#SBATCH --job-name=diffinfinite
#SBATCH --account=a100free
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100-2g-10gb:1
#SBATCH --ntasks=8
#SBATCH --time=48:00:00  # Time limit (HH:MM:SS)
#SBATCH --output=slurm/out_sat.out  # Output file
#SBATCH --error=slurm/sat.err  # Error file
#SBATCH --mail-type=END,FAIL  # Email you when the job finishes or fails
#SBATCH --mail-user=BRGOLI005@myuct.ac.za # Email address to send to
#SBATCH --mem-per-cpu=8000

CUDA_VISIBLE_DEVICES=$(ncvd)

module load python/miniconda3-py310
conda create -y -n diffinf-a100 -f environment.yaml