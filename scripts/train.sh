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
source activate terrain-a100

# https://stackoverflow.com/questions/75921380/python-segmentation-fault-in-interactive-mode
export LANGUAGE=UTF-8
export LC_ALL=en_US.UTF-8
export LANG=UTF-8
export LC_CTYPE=en_US.UTF-8
export LANG=en_US.UTF-8
export LC_COLLATE=$LANG
export LC_CTYPE=$LANG
export LC_MESSAGES=$LANG
export LC_MONETARY=$LANG
export LC_NUMERIC=$LANG
export LC_TIME=$LANG
export LC_ALL=$LANG

export WANDB_API_KEY=$(cat wandb_key)

export WANDB__SERVICE_WAIT=300

host=$(hostname)

echo $host

if [[ $host == *"uct"* ]]; then
    output_dir="/scratch/brgoli005/models/planet"
    data_dir="/scratch/brgoli005/data/"
    batch_size=4
else
    output_dir="/mnt/e/brgoli005/planet"
    data_dir="./planetAI/data/"
    batch_size=2
fi

python -m train \
 --batch_size $batch_size \
 --num_workers $batch_size \
 --results_folder $output_dir/diffinfinite \
 --data_folder $data_dir \ 


conda deactivate