#!/bin/bash
set -eux

cd $(dirname $0)

uv run python train_flow_matching.py \
    --data_path="./data/" \
    --results_dir="./results/$(date '+%Y%m%d_%H%M%S')/"
