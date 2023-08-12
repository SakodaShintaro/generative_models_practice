#!/bin/bash
set -eux

TARGET_DIR=$(readlink -f $1)

cd $(dirname $0)/../
cmake -S . -B build
cmake --build build

./build/vae_cpp $TARGET_DIR
