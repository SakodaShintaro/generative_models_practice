#!/bin/bash
set -eux

cd $(dirname $0)

python3 train.py \
    --dataset="cifar10" \
    --image-size=32 \
    --data-path="./data/"
