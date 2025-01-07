#!/bin/bash
set -eux

cd $(dirname $0)

docker build \
    --build-arg USER_NAME=$(whoami) \
    --build-arg USER_UID=$(id -u) \
    --build-arg USER_GID=$(id -g) \
    -t ${1} .

docker run --gpus all \
            -e DISPLAY=$DISPLAY \
            -v /tmp/.X11-unix:/tmp/.X11-unix \
            -v $HOME/data:$HOME/data/ \
            -v $HOME/work:$HOME/work/ \
            -v $HOME/.cache/:$HOME/.cache/ \
            -v /media:/media \
            -p 7007:7007 \
            -it \
            --ipc=host \
            --privileged \
            ${1} \
            /bin/bash
