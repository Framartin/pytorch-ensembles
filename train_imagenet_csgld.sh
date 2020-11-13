#!/bin/bash -l
# DEBUG
#SBATCH --time=0-01:00:00 # 1 hours
#SBATCH --qos=besteffort
#SBATCH --partition=gpu # Use the batch partition reserved for passive jobs
#SBATCH -J cSGLDImageNet  # Set the job name
#SBATCH -N 1              # 1 computing nodes
#SBATCH -n 1              # 1 tasks
#SBATCH -c 16             # 16 cores per task
#SBATCH --gpus 1          # 4 GPU per tasks
#SBATCH -C volta          # fix type of GPU to compare runtime
#SBATCH -o "log/run_train_imagenet_csgld_%j.log"
#SBATCH --mail-type=end,fail

echo
echo "cSGLD assumes a multi-GPU training, in our case 1 sample tooks 15-17 hours (45 epoch) on 4 v100."
echo

command -v module >/dev/null 2>&1 && module load lang/Python
source ../venv/bin/activate
set -x

DIR="models/ImageNet/resnet50/cSGLD_cycles15_savespercycle12_it1"
DATAPATH="/work/projects/bigdata_sets/ImageNet/ILSVRC2012/raw-data/"
LR=0.1
CYCLES=10
SAMPLES_PER_CYCLE=3

# 1 node with 4 GPUs and 64 cpus (will use as much GPUs available on the node)
python train_imagenet_csgld.py --data $DATAPATH --no-normalization --arch resnet50 \
  --export-dir $DIR --workers 64 \
  --lr $LR --max_lr $LR --print-freq 400 --dist-url tcp://127.0.0.1:5552 --multiprocessing-distributed --world-size 1 --rank 0 \
  --cycles $CYCLES --cycle_epochs 45 --samples-per-cycle $SAMPLES_PER_CYCLE --noise-epochs $SAMPLES_PER_CYCLE

# no fixed seed to speed up
