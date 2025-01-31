#!/bin/bash
#SBATCH --job-name=surgical_job          
#SBATCH --gres=gpu:a100:1                      # 申请 1 个 GPU
#SBATCH --time=20:00:00                   # 时间限制
#SBATCH --output=./logs/work_output_%j.txt # 标准输出日志
#SBATCH --error=./logs/error_%j.log                 # 错误日志
#SBATCH --partition=a100               # 指定分区
#SBATCH --export=ALL,PYTHONUNBUFFERED=1

# 检查模块加载器是否存在并加载必要模块
source /apps/modules/5.1.1/init/bash
module load cuda/11.8.0

# 激活 Conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate myenv

# 创建日志目录
mkdir -p logs

# 查看 GPU 信息
nvidia-smi

# 运行 Python 脚本
python surgical_phase_classification.py \
  --model_dir /home/hpc/iwb9/iwb9102h/model_cache/models--google--gemma-2-2b-it/snapshots/299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8 \
  --data_folder /home/hpc/iwb9/iwb9102h/surgical_phase_code/datadoctor \
  --device cuda \
  --sample_size 200








