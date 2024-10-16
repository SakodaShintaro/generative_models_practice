#!/bin/bash
set -eux

cd $(dirname $0)

python3 train_flow_matching.py \
    --dataset="stl10" \
    --epochs=3000 \
    --image_size=64 \
    --batch_size=500 \
    --learning_rate=1e-4 \
    --data_path="./data/" \
    --results_dir="./results/$(date '+%Y%m%d_%H%M%S')/"
