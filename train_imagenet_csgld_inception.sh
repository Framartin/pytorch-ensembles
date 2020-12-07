#!/bin/bash -l
#SBATCH --time=0-02:00:00 # 9 days
#SBATCH --qos=long
#SBATCH --partition=gpu   # Use the batch partition reserved for passive jobs
#SBATCH -J cSGLDinception  # Set the job name
#SBATCH -N 1              # 1 computing nodes
#SBATCH -n 1              # 1 tasks
#SBATCH -c 16             # 16 cores per task
#SBATCH --gpus 4          # 4 GPU per tasks
#SBATCH -C volta          # fix type of GPU to compare runtime
#SBATCH -o "log/run_train_imagenet_csgld_inception_%j.log"
#SBATCH --mail-type=end,fail

echo
echo "cSGLD assumes a multi-GPU training."
echo

command -v module >/dev/null 2>&1 && module load lang/Python system/CUDA
source ../venv/bin/activate
set -x

DATAPATH="/work/projects/bigdata_sets/ImageNet/ILSVRC2012/raw-data/"
ARCH="inception_v3"
DIST_URL="file://${SCRATCH}tmp/torchfilestore${ARCH}"  # becareful: should be unique per script call
rm -f ${SCRATCH}tmp/torchfilestore # delete previous file
LR=0.1
CYCLES=5
SAMPLES_PER_CYCLE=3
BATCH_SIZE=256
WORKERS=16
PRINT_FREQ=400
#debug:
#BATCH_SIZE=32
#PRINT_FREQ=10
#WORKERS=8
DIR="../models/ImageNet/${ARCH}/cSGLD_cycles${CYCLES}_samples${SAMPLES_PER_CYCLE}_bs${BATCH_SIZE}"

date

# 1 node with 4 GPUs and 16 cpus (will use as much GPUs available on the node)
python -u train_imagenet_csgld.py --data $DATAPATH --no-normalization --arch $ARCH \
  --export-dir $DIR --workers $WORKERS --batch-size $BATCH_SIZE \
  --lr $LR --max-lr $LR --print-freq $PRINT_FREQ --dist-url $DIST_URL --multiprocessing-distributed --world-size 1 --rank 0 \
  --cycles $CYCLES --cycle-epochs 45 --samples-per-cycle $SAMPLES_PER_CYCLE --noise-epochs $SAMPLES_PER_CYCLE

# no fixed seed to speed up

date