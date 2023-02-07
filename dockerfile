ARG CONFIGURATION="release"

FROM nvidia/cuda:12.0.1-devel-ubuntu22.04 as builder
ARG CONFIGURATION

#
WORKDIR /workspace

# Copy sources
COPY examples/ examples
COPY src/ src
COPY ThirdParty/ ThirdParty
COPY makefile makefile

# make
RUN make $CONFIGURATION

# runner
FROM nvidia/cuda:12.0.1-base-ubuntu22.04
ARG CONFIGURATION

RUN groupadd cuda && useradd -m -d /app -g cuda cuda
USER cuda

WORKDIR /app
COPY --chown=cuda:cuda --from=builder /workspace/x64/$CONFIGURATION/bin /app/
ENTRYPOINT [ "/app/CudaKeeloq", "--help" ]
