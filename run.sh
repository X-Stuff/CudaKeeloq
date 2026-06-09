#!/bin/sh

CONTAINER="${CONTAINER:-cudakeeloq}"
TAG="${TAG:-local}"
DOCKER_ARGS="${DOCKER_ARGS:-}"

docker run --rm -it --init --gpus=all --device /dev/dxg:/dev/dxg ${DOCKER_ARGS} $CONTAINER:$TAG $@

