#!/bin/bash

#SBATCH --job-name=run_pi0           # 作业名
#SBATCH --partition=gpuB                # 使用 gpuB 分区
#SBATCH --nodes=1                       # 只使用一个节点
#SBATCH --ntasks-per-node=1             # 一个训练任务（DDP rank 0）
#SBATCH --cpus-per-task=12              # 根据卡配置调整，一般 12 核对一张 A100 是合理的
#SBATCH --gres=gpu:1                    # 请求 1 张 GPU
#SBATCH -o %j.out                       # 标准输出
#SBATCH -e %j.err                       # 错误输出
#SBATCH --mail-type=BEGIN,END,FAIL      # 可以修改，任务开始，结束，失败时邮件通知
#SBATCH --mail-user=xxxx@seu.edu.cn     # 可以修改，邮件通知邮箱      

# 加载环境模块
module load anaconda3-2024.10-1

__conda_setup="$('/seu_share/apps/anaconda3-2024.10-1/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/seu_share/apps/anaconda3-2024.10-1/etc/profile.d/conda.sh" ]; then
        . "/seu_share/apps/anaconda3-2024.10-1/etc/profile.d/conda.sh"
    else
        export PATH="/seu_share/apps/anaconda3-2024.10-1/bin:$PATH"
    fi
fi
unset __conda_setup

conda activate pi0l

export WANDB_MODE=offline

cd /seu_share2/home/linli/213221090/ai-learning/pi0/code

export MODEL_ROOT=/seu_nvme/home/linli/213221090/MyModels
export DATA_ROOT=/seu_nvme/home/linli/213221090/MyDatasets
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python run_local_pi0_libero.py \
  --policy-path "$MODEL_ROOT/lerobot/pi0_libero_finetuned" \
  --tokenizer-path "$MODEL_ROOT/google/paligemma-3b-pt-224" \
  --local-files-only \
  --suite libero_object \
  --task-id 0 \
  --episodes 1 \
  --steps 200 \
  --trace \
  --debug
