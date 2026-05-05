ARG CONFIGURATION="release"
ARG CUDA_MAJOR=13
ARG CUDA_MINOR=2
ARG CUDA_PATCH=0

FROM nvidia/cuda:${CUDA_MAJOR}.${CUDA_MINOR}.${CUDA_PATCH}-devel-ubuntu22.04 AS builder
ARG CONFIGURATION

WORKDIR /workspace

# Copy sources
COPY examples/ examples
COPY src/ src
COPY ThirdParty/ ThirdParty
COPY makefile makefile

# Build both binaries (app + test runner).
RUN make ${CONFIGURATION}

# Runner: ship both binaries. Default entrypoint stays the main app.
FROM nvidia/cuda:${CUDA_MAJOR}.${CUDA_MINOR}.${CUDA_PATCH}-base-ubuntu22.04
ARG CONFIGURATION

RUN groupadd cuda && useradd -m -d /app -g cuda cuda
USER cuda

WORKDIR /app
COPY --chown=cuda:cuda --from=builder /workspace/x64/${CONFIGURATION}/bin /app/
ENTRYPOINT [ "/app/CudaKeeloq" ]
CMD [ "--help" ]

# To run the test suite instead of the app:
#   docker run --rm --gpus=all cudakeeloq:local --entrypoint /app/CudaKeeloqTests
# Or rebuild with --build-arg CONFIGURATION=debug for -G device debug info.
