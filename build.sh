#!/bin/sh

CONTAINER="${CONTAINER:-cudakeeloq}"
TAG="${TAG:-local}"

docker build . -t $CONTAINER:$TAG

