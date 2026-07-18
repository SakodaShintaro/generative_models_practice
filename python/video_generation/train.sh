#!/bin/bash
set -eux

cd $(dirname $0)

# Temporal sequence model under test: attention | gru | gated_deltanet | ttt
TEMPORAL="${1:-ttt}"

uv run python train.py \
    --data_root="/media/sakoda/samsung_4t/bench2drive" \
    --results_dir="./results/${TEMPORAL}_$(date '+%Y%m%d_%H%M%S')/" \
    --temporal="${TEMPORAL}"
