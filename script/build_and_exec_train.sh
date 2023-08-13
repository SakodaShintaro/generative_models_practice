#!/bin/bash
set -eux

TARGET_DIR=$(readlink -f $1)

cd $(dirname $0)/../

./script/apply_clang_format.sh

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

./build/vae_cpp train $TARGET_DIR
