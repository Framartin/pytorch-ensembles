#!/bin/bash -l

# launch with:
# bash train_sse_mcmc_dgx.sh >>log/cifar10/train_sse_mcmc_dgx.log 2>&1

source /opt/miniconda/bin/activate
conda activate pytorch

set -x

#specify GPU
export CUDA_VISIBLE_DEVICES=1

bash train_sse_mcmc.sh CIFAR10 PreResNet164 1 ../models ../data cSGLD
# 250 epochs in 323 minutes

bash train_sse_mcmc.sh CIFAR10 VGG16BN 1 ../models ../data cSGLD

bash train_sse_mcmc.sh CIFAR10 VGG19BN 1 ../models ../data cSGLD

#bash train_sse_mcmc.sh CIFAR10 WideResNet28x10 1 ../models ../data cSGLD
