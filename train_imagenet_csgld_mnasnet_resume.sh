#!/bin/bash -l
#should take XX days
# launch with:
# bash train_imagenet_csgld_mnasnet_resume.sh >>log/run_train_imagenet_csgld_mnasnet_resume.log 2>&1

echo
echo "single GPU training - manual scheduling"
echo
echp "! try to resume with lower weight decay"

source /opt/miniconda/bin/activate
conda activate pytorch

set -x

#specify GPU
CUDA_VISIBLE_DEVICES=2


DATAPATH="../../data/ILSVRC2012"
ARCH="mnasnet1_0"
LR=0.1
WEIGHT_DECAY=1e-5
CYCLES=3
SAMPLES_PER_CYCLE=3
BATCH_SIZE=64
WORKERS=10
PRINT_FREQ=400
DIR="../models/ImageNet/${ARCH}/cSGLD_cycles${CYCLES}_samples${SAMPLES_PER_CYCLE}_bs${BATCH_SIZE}_lr${LR}_wd${WEIGHT_DECAY}"


date


python -u train_imagenet_csgld.py --data $DATAPATH --no-normalization --arch $ARCH \
  --export-dir $DIR --workers $WORKERS --batch-size $BATCH_SIZE \
  --lr $LR --max-lr $LR --print-freq $PRINT_FREQ --world-size 1 \
  --cycles $CYCLES --cycle-epochs 45 --samples-per-cycle $SAMPLES_PER_CYCLE --noise-epochs $SAMPLES_PER_CYCLE \
  --gpu $CUDA_VISIBLE_DEVICES \
  --weight-decay $WEIGHT_DECAY \
  --resume "../models/ImageNet/${ARCH}/cSGLD_cycles3_samples3_bs64/ImageNet-cSGLD_mnasnet1_0_00044.pt.tar"

date