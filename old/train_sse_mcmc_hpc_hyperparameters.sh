#!/bin/bash -l
#SBATCH --time=0-17:00:00 # 17 hours
#SBATCH --partition=gpu # Use the batch partition reserved for passive jobs
#SBATCH -J TrainCycl      # Set the job name
#SBATCH -N 1              # 1 computing nodes
#SBATCH -n 1              # 2 tasks
#SBATCH -c 4              # 2 cores per task
#SBATCH --gpus 1          # 1 GPU per tasks
#SBATCH -C volta          # fix type of GPU to compare runtime
#SBATCH -o "log/run_train_csgld_HP_%j.log"
#SBATCH --mail-type=end,fail

command -v module >/dev/null 2>&1 && module load lang/Python

source ../venv/bin/activate
set -x


bash ./train_sse_mcmc.sh CIFAR10 PreResNet110 1 ../models ../data cSGLD12cycles
# on single GPU PreResNet110: 12 cycles of 62 epochs with 12 saves on 14 hours
