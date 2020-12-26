#!/bin/bash -l
# launch with:
# bash train_imagenet_csgld_efficientnet_resume1_dgx.sh >>log/run_train_imagenet_csgld_efficientnet_resume1_dgx.log 2>&1

echo
echo "single GPU training"
echo

command -v module >/dev/null 2>&1 && module load lang/Python system/CUDA
source ../venv/bin/activate
set -x

#specify GPU
CUDA_VISIBLE_DEVICES=0


DATAPATH="../../data/ILSVRC2012"
ARCH="efficientnet_b0"
LR=0.1
CYCLES=3
SAMPLES_PER_CYCLE=3
BATCH_SIZE=256
WORKERS=10
PRINT_FREQ=400
#DIR="../models/ImageNet/${ARCH}/cSGLD_cycles${CYCLES}_samples${SAMPLES_PER_CYCLE}_bs${BATCH_SIZE}_resume1"
DIR="../models/ImageNet/efficientnet_b0/cSGLD_cycles3_samples3_bs64_resume1"

date

python -u train_imagenet_csgld.py --data $DATAPATH --no-normalization --arch $ARCH \
  --export-dir $DIR --workers $WORKERS --batch-size $BATCH_SIZE \
  --lr $LR --max-lr $LR --print-freq $PRINT_FREQ --world-size 1 \
  --cycles $CYCLES --cycle-epochs 45 --samples-per-cycle $SAMPLES_PER_CYCLE --noise-epochs $SAMPLES_PER_CYCLE \
  --gpu $CUDA_VISIBLE_DEVICES \
  --resume "${DIR}/ImageNet-cSGLD_efficientnet_b0_00089.pt.tar"

date