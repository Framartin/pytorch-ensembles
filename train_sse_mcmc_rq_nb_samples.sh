#!/bin/bash -l

# launch with:
# bash train_sse_mcmc_rq_nb_samples.sh >>log/cifar10/train_sse_mcmc_rq_nb_samples.log 2>&1

source /opt/miniconda/bin/activate
conda activate pytorch

set -x

#specify GPU
export CUDA_VISIBLE_DEVICES=3

DATASET="CIFAR10"
ARCH="PreResNet110"
ITER=1
BASE_DIR="../models"
DATA_PATH="../data"

# original config:
#  INJECT_NOISE_OR_NOT="--inject_noise"
#  MAX_LR=0.5
#  CYCLE_EPOCHS=50
#  CYCLE_SAVES=3
#  CYCLES=34
#  NOISE_EPOCHS=3
INJECT_NOISE_OR_NOT="--inject_noise"
MAX_LR=0.5
CYCLE_EPOCHS=57
CYCLE_SAVES=10
CYCLES=5
NOISE_EPOCHS=10

WD=3e-4

python sse_mcmc_train.py $INJECT_NOISE_OR_NOT \
    --dir="${BASE_DIR}/${DATASET}/${ARCH}/cSGLD_cycles${CYCLES}_savespercycle${CYCLE_SAVES}_it${ITER}" \
    --model="$ARCH" --dataset="$DATASET" --noise_epochs=$NOISE_EPOCHS --data_path="$DATA_PATH" \
    --alpha=1 --cycles=$CYCLES --iter="$ITER" \
    --cycle_epochs=$CYCLE_EPOCHS --cycle_saves=$CYCLE_SAVES --max_lr=$MAX_LR --wd=$WD \
    --device_id 0 --transform="NoNormalization"
