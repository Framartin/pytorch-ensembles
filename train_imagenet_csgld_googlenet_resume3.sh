#!/bin/bash -l
# launch with:
# bash train_imagenet_csgld_googlenet_resume3.sh >>log/run_train_imagenet_csgld_googlenet_resume3.log 2>&1

echo
echo "single GPU training - manual scheduling"
echo

source /opt/miniconda/bin/activate
conda activate pytorch

set -x

#specify GPU
CUDA_VISIBLE_DEVICES=0

DATAPATH="../../data/ILSVRC2012"
ARCH="googlenet"
LR=0.001
WEIGHT_DECAY=0.00004
CYCLES=3
SAMPLES_PER_CYCLE=3
BATCH_SIZE=256
WORKERS=10
PRINT_FREQ=400
DIR="../models/ImageNet/${ARCH}/cSGLD_cycles${CYCLES}_samples${SAMPLES_PER_CYCLE}_bs${BATCH_SIZE}_lr${LR}_wd${WEIGHT_DECAY}"


date
# resume after nan loss

python -u train_imagenet_csgld.py --data $DATAPATH --no-normalization --arch $ARCH \
  --export-dir $DIR --workers $WORKERS --batch-size $BATCH_SIZE \
  --lr $LR --max-lr $LR --print-freq $PRINT_FREQ --world-size 1 \
  --cycles $CYCLES --cycle-epochs 45 --samples-per-cycle $SAMPLES_PER_CYCLE --noise-epochs $SAMPLES_PER_CYCLE \
  --gpu $CUDA_VISIBLE_DEVICES \
  --weight-decay $WEIGHT_DECAY  \
  --resume "../models/ImageNet/googlenet/cSGLD_cycles3_samples3_bs256_lr0.1_wd0.00004/ImageNet-cSGLD_googlenet_00044.pt.tar"

date