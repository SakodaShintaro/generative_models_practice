#!/bin/bash
set -eux

cd $(dirname $0)

python3 train.py \
    --dataset="cifar10" \
    --image_size=32 \
    --data_path="./data/" \
    --results_dir="./results/$(date '+%Y%m%d_%H%M%S')/"
