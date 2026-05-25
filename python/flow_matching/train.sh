#!/bin/bash
set -eux

cd $(dirname $0)

uv run python train_flow_matching.py \
    --epochs=3000 \
    --batch_size=500 \
    --learning_rate=1e-4 \
    --data_path="./data/" \
    --results_dir="./results/$(date '+%Y%m%d_%H%M%S')/"
