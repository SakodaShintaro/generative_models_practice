#!/bin/bash
set -eux

cd $(dirname $0)/../..

TORCH_VERSION=2.0.1
CUDA_VERSION=117
CUDA_STR=cu${CUDA_VERSION}
URL_FILE_NAME=libtorch-cxx11-abi-shared-with-deps-${TORCH_VERSION}%2B${CUDA_STR}.zip
FILE_NAME=libtorch-cxx11-abi-shared-with-deps-${TORCH_VERSION}+${CUDA_STR}.zip

wget https://download.pytorch.org/libtorch/${CUDA_STR}/${URL_FILE_NAME}
unzip -q ${FILE_NAME} -d libtorch-tmp
mv libtorch-tmp/libtorch libtorch-${TORCH_VERSION}
rm ${FILE_NAME}
rmdir libtorch-tmp
