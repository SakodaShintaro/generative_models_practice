#!/bin/bash
set -eux

OUTPUT_DIR=$(readlink -f $1)

cd $(dirname $0)/../

./script/apply_clang_format.sh

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

rm -rf $OUTPUT_DIR
mkdir -p $OUTPUT_DIR
./build/vae_cpp generate $OUTPUT_DIR
