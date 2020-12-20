#!/bin/bash -l
# launch with:
# bash train_imagenet_csgld_efficientnet2.sh >>log/run_train_imagenet_csgld_efficientnet2.log 2>&1

echo
echo "single GPU training - manual scheduling"
echo

source /opt/miniconda/bin/activate
conda activate pytorch

set -x

#specify GPU
CUDA_VISIBLE_DEVICES=2


echo
echo "single GPU training"
echo

command -v module >/dev/null 2>&1 && module load lang/Python system/CUDA
source ../venv/bin/activate
set -x

DATAPATH="/work/projects/bigdata_sets/ImageNet/ILSVRC2012/raw-data/"
ARCH="efficientnet_b0"

# HPs closer to https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/main.py
LR=0.016
#LR=0.256 # in the paper (for 8 GPUs?)
WEIGHT_DECAY=1e-5
CYCLES=3
SAMPLES_PER_CYCLE=3
BATCH_SIZE=256
WORKERS=10
PRINT_FREQ=400
DIR="../models/ImageNet/${ARCH}/cSGLD_cycles${CYCLES}_samples${SAMPLES_PER_CYCLE}_bs${BATCH_SIZE}_lr${LR}_wd${WEIGHT_DECAY}"

date

python -u train_imagenet_csgld.py --data $DATAPATH --no-normalization --arch $ARCH \
  --export-dir $DIR --workers $WORKERS --batch-size $BATCH_SIZE \
  --lr $LR --max-lr $LR --print-freq $PRINT_FREQ --world-size 1 \
  --cycles $CYCLES --cycle-epochs 45 --samples-per-cycle $SAMPLES_PER_CYCLE --noise-epochs $SAMPLES_PER_CYCLE \
  --gpu $CUDA_VISIBLE_DEVICES \
  --weight-decay $WEIGHT_DECAY

date