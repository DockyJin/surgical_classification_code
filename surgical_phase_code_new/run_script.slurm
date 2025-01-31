#!/bin/bash
#SBATCH --job-name=surgical_job          
#SBATCH --gres=gpu:a100:1
#SBATCH --time=20:00:00
#SBATCH --output=./logs/work_output_%j.txt
#SBATCH --error=./logs/error_%j.log          
#SBATCH --partition=a100        
#SBATCH --export=ALL,PYTHONUNBUFFERED=1

source /apps/modules/5.1.1/init/bash
module load cuda/11.8.0

source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv

mkdir -p logs

nvidia-smi

python surgical_phase_classification.py \
  --model_dir /home/hpc/iwb9/iwb9102h/model_cache/models--google--gemma-2-2b-it/snapshots/299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8 \
  --data_folder /home/hpc/iwb9/iwb9102h/surgical_phase_code/datadoctor \
  --device cuda \
  --sample_size 200








