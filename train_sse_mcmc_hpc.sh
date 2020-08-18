#!/bin/bash -l
#SBATCH --time=0-14:00:00 # 14 hours
#SBATCH --partition=gpu # Use the batch partition reserved for passive jobs
#SBATCH --qos=qos-gpu
#SBATCH -J TrainCycl      # Set the job name
#SBATCH -N 1              # 1 computing nodes
#SBATCH -n 2              # 2 tasks
#SBATCH -c 2              # 2 cores per task
#SBATCH --gpus 1          # 1 GPU per tasks
#SBATCH -C volta          # fix type of GPU to compare runtime
#SBATCH -o "log/run_train_csgld_%j.log"
#SBATCH --mail-type=end,fail


set -x
module load lang/Python

source ../venv/bin/activate

# VGG16 doesn't work
#bash ./train_sse_mcmc.sh CIFAR10 VGG16 1 ../models ../data cSGLD

bash ./train_sse_mcmc.sh CIFAR10 PreResNet110 1 ../models ../data cSGLD
# on single GPU PreResNet110: 30 cycles of 50 epochs with 3 saves on 7 hours (524 epochs on 7 hours)