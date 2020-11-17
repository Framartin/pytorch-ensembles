#!/bin/bash -l
# DEBUG
#SBATCH --time=0-00:15:00 # 15 minutes
#SBATCH --qos=besteffort
#SBATCH --partition=gpu   # Use the batch partition reserved for passive jobs
#SBATCH -J cSGLDImageNet  # Set the job name
#SBATCH -N 1              # 1 computing nodes
#SBATCH -n 1              # 1 tasks
#SBATCH -c 8              # 8 cores per task
#SBATCH --gpus 1          # 1 GPU per tasks
#SBATCH -C volta          # fix type of GPU to compare runtime
#SBATCH -o "log/run_train_imagenet_csgld_debug_%j.log"
#SBATCH --mail-type=end,fail

echo
echo "cSGLD assumes a multi-GPU training, in our case 1 cycle tooks 15-17 hours (45 epochs) on 4 v100."
echo

command -v module >/dev/null 2>&1 && module load lang/Python system/CUDA
source ../venv/bin/activate
set -x

DATAPATH="/work/projects/bigdata_sets/ImageNet/ILSVRC2012/raw-data/"
DIST_URL="file://${SCRATCH}tmp/torchfilestore_debug"  # becareful: should be unique per script call
rm -f ${SCRATCH}tmp/torchfilestore_debug # delete previous file
LR=0.1
CYCLES=1
SAMPLES_PER_CYCLE=1
BATCH_SIZE=64
WORKERS=8
PRINT_FREQ=10
DIR="../models/ImageNet/resnet50/DEBUG_cSGLD_cycles${CYCLES}_samples${SAMPLES_PER_CYCLE}_bs${BATCH_SIZE}"

date

# 1 node with 4 GPUs and 16 cpus (will use as much GPUs available on the node)
python -u train_imagenet_csgld.py --data $DATAPATH --no-normalization --arch resnet50 \
  --export-dir $DIR --workers $WORKERS --batch-size $BATCH_SIZE \
  --lr $LR --max-lr $LR --print-freq $PRINT_FREQ --dist-url $DIST_URL --multiprocessing-distributed --world-size 1 --rank 0 \
  --cycles $CYCLES --cycle-epochs 1 --samples-per-cycle $SAMPLES_PER_CYCLE --noise-epochs $SAMPLES_PER_CYCLE \
  --debug

# no fixed seed to speed up

date