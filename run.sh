#!/bin/sh

CONTAINER="${CONTAINER:-cudakeeloq}"
TAG="${TAG:-local}"

docker run --rm -it --init --gpus=all $CONTAINER:$TAG $@

