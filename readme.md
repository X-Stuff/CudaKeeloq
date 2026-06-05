# CudaKeeloq

CudaKeeloq is a CUDA-accelerated KeeLoq bruteforcer. It can test manufacturer
keys, seed values, dictionary entries, and constrained key spaces using NVIDIA
GPUs.

## Disclaimer

A 64-bit KeeLoq manufacturer key has `18,446,744,073,709,551,616` possible
values. Even at 1 billion keys per second, a full key search takes about 584
years. This project is useful for research, validation, and constrained search
spaces, but a raw full-range attack is not practical.

Use this only on systems and captures you are allowed to analyze.

## Version history

### 0.2.0

* Updated the build to CUDA 13.2.
* Refactored the bruteforce pipeline, CUDA device helpers, kernel input handling,
  and result handling.
* Added benchmark tooling that sweeps CUDA block/thread configurations and
  reports the best throughput for each tested workload.
* Added a separate `CudaKeeloqTests` test runner and expanded CLI, CUDA smoke,
  generator, filter, alphabet, and KeeLoq tests.
* Added named CLI values for bruteforce modes and learning types. Numeric values
  still work.
* Added input transforms for reversed manufacturer keys (`--check-rev`, enabled
  by default) and XOR variants over the fixed part, hopping part, and decrypted
  hopping part (`--check-xorfix`, `--check-xorhop`, `--check-xordec`).
* Added inverted algorithm checks for Normal, Secure, and FAAC learning types
  (`--check-inv-algs`, enabled by default).
* Added Serial1, Serial2, and Serial3 learning calculations.
* Added Xor bruteforce mode for fixed manufacturer keys and unknown XOR values.
* Simplified the matching path around the normal three-capture workflow.
* Improved Docker support. The image now builds the app and test runner, ships a
  runnable entrypoint, and defaults to `--help`.

### 0.1.2

* Fixed the Dockerfile and added a default `CMD`.
* Updated CUDA to 12.2.

### 0.1.1

* Added seed bruteforce mode.
* Added seed support in text dictionaries.
* Fixed minor bugs and did internal cleanup.
* Improved performance by about 1-5%.

### 0.1.0

* Initial public release.

## Capabilities

* **Dictionary** - reads manufacturer keys from text or binary dictionaries.
  Text dictionaries may include seeds.
* **Simple** - increments manufacturer keys by `+1` from `--start`.
* **Filtered** - simple bruteforce with include/exclude key filters.
* **Alphabet** - generates keys from a fixed set of allowed bytes.
* **Pattern** - generates keys from per-byte constants, ranges, alternatives,
  wildcards, or alphabets.
* **Seed** - bruteforces a 32-bit seed for a known manufacturer key.
* **Xor** - bruteforces 32-bit XOR values for a known manufacturer key.

## Limitations

* Full 64-bit manufacturer-key bruteforce is usually not practical.
* The normal workflow expects three captures from the same transmitter button.
  Fewer captures are likely to produce too many phantom matches.
* The app does not brute a 64-bit manufacturer key and a 32-bit seed or XOR
  value as one combined search space. Use a fixed seed/XOR value while
  bruteforcing keys, or use a fixed manufacturer key while bruteforcing seeds
  or XOR values.
* Binary dictionary window attacks are not implemented. If you need shifted
  dictionary windows, prepare shifted dictionary files yourself.

## Requirements

### Runtime

* NVIDIA GPU with CUDA support.
* NVIDIA driver compatible with CUDA 13.2.
* Enough GPU memory for the selected CUDA launch configuration.
* Linux Docker runs require NVIDIA Container Toolkit.

### Windows build

* CUDA Toolkit 13.2.
* Microsoft Visual Studio 2022 with the v143 toolset.

Open `CudaKeeloq.vcxproj` in Visual Studio and build the desired configuration.
`CudaKeeloqTests.vcxproj` builds the test runner.

> NOTE: Release build takes significant amount of time, around 5 minutes, due to enormous amount of kernels permutations and device link time optimizations.

### Linux build

Requirements:

* CUDA Toolkit 13.2.
* `make`
* `g++`

```sh
make release
```

For a debug build:

```sh
make debug
```

Other useful targets:

```sh
make profile
make app CONFIG=release
make tests CONFIG=release
make clean
```

The binaries are written under `.build/x64/<config>/linux/bin/`.

## Docker usage

Docker is optional. It is useful when you want a reproducible CUDA 13.2 build
environment or a ready-to-run image.

Build the image:

```sh
./build.sh
```

This builds `cudakeeloq:local`. You can override the image name:

```sh
CONTAINER=my-keeloq TAG=dev ./build.sh
```

Run the app through the helper script:

```sh
./run.sh --help
```

Example bruteforce run:

```sh
./run.sh \
  --inputs=0xC65D52A0A81FD504,0xCCA9B335A81FD504,0xE0DA7372A81FD504 \
  --mode=simple \
  --start=0x9876543210 \
  --count=1000000
```

The helper runs:

```sh
docker run --rm -it --init --gpus=all --device /dev/dxg:/dev/dxg cudakeeloq:local <args>
```

On WSL, `--device /dev/dxg:/dev/dxg` is required for the container to access
the GPU. On native Linux, remove that option if `/dev/dxg` does not exist.

To use files that are not already copied into the image, mount them with
`DOCKER_ARGS`:

```sh
DOCKER_ARGS="-v $PWD/data:/data" ./run.sh \
  --inputs=0xC65D52A0A81FD504,0xCCA9B335A81FD504,0xE0DA7372A81FD504 \
  --mode=dictionary \
  --word-dict=/data/keys.txt
```

Run the test binary from the Docker image:

```sh
docker run --rm --gpus=all --entrypoint /app/CudaKeeloqTests cudakeeloq:local
```

## Benchmarking

Use the benchmark mode to find the fastest CUDA launch settings for your GPU:

```sh
./run.sh --benchmark=true
```

The benchmark tries different block/thread counts and both single-learning and
multi-learning kernel modes. For each workload it prints the best result, for
example:

```text
Best result: Multiple learnings mode in kernel, 8192 blocks, 256 threads - 1234.567 million keys/s
```

Use the reported values with:

```sh
./run.sh \
  --inputs=<input1>,<input2>,<input3> \
  --mode=simple \
  --start=<first-key> \
  --count=<count> \
  --cuda-blocks=<best-blocks> \
  --cuda-threads=<best-threads>
```

Press `Esc` during benchmarking to skip the current benchmark section.

## Performance

These values depend on the GPU, driver, CUDA version, build configuration, and
selected learning types. Use `--benchmark=true` to measure your own setup.

Shared configuration:

* GPU: laptop RTX 3080 Ti.
* CUDA: 13.2.
* Build: Release.

| Result | Native Windows | WSL | Docker in WSL |
| --- | --- | --- | --- |
| `ALL` learning types with seed |  |  |  |
| `ALL` learning types without seed |  |  |  |
| `Simple` |  |  |  |
| `Normal` |  |  |  |
| `Secure` |  |  |  |
| `FAAC` |  |  |  |
| `Xor` |  |  |  |

## Captures and inputs

`--inputs` expects three captured KeeLoq OTA packets as 64-bit hex values:

```text
0x<HOPPING_32><FIXED_32>
```

Capture the same transmitter button three times. The fixed part should match,
and the decoded counters should increase. Fewer captures are not useful for
normal operation because they are much more likely to produce false matches.

## Examples

### Simple bruteforce

```sh
./CudaKeeloq \
  --inputs=0xC65D52A0A81FD504,0xCCA9B335A81FD504,0xE0DA7372A81FD504 \
  --mode=simple \
  --start=0x9876543210 \
  --count=1000000
```

Checks 1 million manufacturer keys starting at `0x9876543210`.

### Alphabet bruteforce

```sh
./CudaKeeloq \
  --inputs=0xC65D52A0A81FD504,0xCCA9B335A81FD504,0xE0DA7372A81FD504 \
  --mode=alphabet \
  --learning-type=Simple \
  --alphabet=examples/alphabet.bin,10:20:30:AA:BB:CC:DD:EE:FF:02:33
```

This starts one attack using `examples/alphabet.bin` and another using the byte
list from the command line.

### Pattern bruteforce

```sh
./CudaKeeloq \
  --inputs=0xC65D52A0A81FD504,0xCCA9B335A81FD504,0xE0DA7372A81FD504 \
  --mode=pattern \
  --pattern=FF:11:*:*:AA-FF:01|10:00:FF,*:*:*:*:AB:CD:EF:00
```

Patterns are written in big-endian byte order. `*` means any byte, `AA-FF` means
a byte range, and `01|10` means either byte.

### Dictionary bruteforce

```sh
./CudaKeeloq \
  --inputs=0xC65D52A0A81FD504,0xCCA9B335A81FD504,0xE0DA7372A81FD504 \
  --mode=dictionary \
  --word-dict=0x0FDE4531BBACAD12,examples/dictionary.words \
  --bin-dict=examples/dictionary.bin \
  --bin-dict-mode=1
```

Text dictionaries contain one key per line. To include a decimal seed in a text
dictionary, use `key:seed`, for example:

```text
0xAABBCCDDEEFF0011:123456
```

### Seed bruteforce

```sh
./CudaKeeloq \
  --inputs=0xC65D52A0A81FD504,0xCCA9B335A81FD504,0xE0DA7372A81FD504 \
  --mode=seed \
  --start=0xAABBCCDDEEFF0011
```

This keeps the manufacturer key fixed and checks the seed space for seeded
learning types such as Secure and FAAC.

### Xor bruteforce

```sh
./CudaKeeloq \
  --inputs=0xC65D52A0A81FD504,0xCCA9B335A81FD504,0xE0DA7372A81FD504 \
  --mode=xor \
  --start=0xAABBCCDDEEFF0011 \
  --seed=0
```

This keeps the manufacturer key fixed and searches XOR values for supported
input transform locations.

## Command line reference

### General

* `--help`, `-h` - print help.
* `--version`, `-v` - print the version.
* `--benchmark=true` - run the benchmark suite instead of a bruteforce run.
* `--inputs=<i1,i2,i3>` - three captured OTA packets.
* `--first-match=true|false` - stop when the first match is found. Default:
  `true`.

### CUDA settings

* `--cuda-blocks=<num>` - CUDA blocks per kernel launch. `0` lets the app choose.
* `--cuda-threads=<num>` - CUDA threads per block. `0` lets the app choose.

Higher values can improve throughput, but they also use more GPU memory. Use
`--benchmark=true` to find good values for your GPU and workload.

### Modes

`--mode` accepts comma-separated names or numeric IDs:

* `0`, `dictionary` - dictionary attack. Uses `--word-dict` and/or `--bin-dict`.
* `1`, `simple` - `+1` manufacturer-key bruteforce.
* `2`, `filtered` - simple bruteforce with filters.
* `3`, `alphabet` - generate keys from one or more alphabets.
* `4`, `pattern` - generate keys from one or more byte patterns.
* `5`, `seed` - brute seeds for a fixed manufacturer key.
* `6`, `xor` - brute XOR values for a fixed manufacturer key.

Several modes can be selected at once, for example:

```sh
--mode=dictionary,simple,pattern
```

Provide all arguments required by every selected mode.

### Key range

* `--start=<value>` - first manufacturer key, or the fixed manufacturer key for
  `seed` and `xor` modes. Default: `0`.
* `--count=<value>` - number of generated values to check. Default:
  `0xFFFFFFFFFFFFFFFF`.
* `--seed=<value>` - fixed seed for learning types that need one. For XOR
  transforms it is also used as the XOR value; in `xor` mode it is the first XOR
  value to test. For text dictionaries, a per-entry seed can also be written as
  `key:seed`.

### Dictionaries

* `--word-dict=<file,key,...>` - text dictionary files or literal hex keys.
* `--bin-dict=<file,...>` - binary dictionary files. Each 8 bytes are read as
  one key.
* `--bin-dict-mode=0` - read binary dictionary keys as-is, big-endian.
* `--bin-dict-mode=1` - read binary dictionary keys reversed, little-endian.
* `--bin-dict-mode=2` - add both byte orders.

### Alphabet and pattern

* `--alphabet=<file,AA:BB:01,...>` - binary alphabet files or colon-separated
  byte lists.
* `--pattern=<pattern,...>` - big-endian byte patterns.

Pattern byte syntax:

* `*` - any byte.
* `0A` or `0x0A` - exact byte.
* `0x10-0x32` - inclusive byte range.
* `33|44|FA` - one of the listed bytes.
* `AL0`, `AL1`, ... - reuse alphabets from `--alphabet` by index.

### Filters

* `--exclude-filter=<flags>` - skip generated keys matching these filter flags.
* `--include-filter=<flags>` - keep only generated keys matching these filter
  flags. This can be slow.

See `src/bruteforce/bruteforce_filters.h` for the filter flag values.

### Learning types and transforms

`--learning-type` accepts comma-separated names or numeric IDs:

* `0`, `Simple`
* `1`, `Normal`
* `2`, `Secure` - requires a seed.
* `3`, `Xor`
* `4`, `FAAC` - requires a seed.
* `5`, `Serial1`
* `6`, `Serial2`
* `7`, `Serial3`
* `ALL` - default.

Related options:

* `--check-rev=true|false` - also check byte-reversed manufacturer keys.
  Default: `true`.
* `--check-xorfix=true|false` - also check XOR applied to the fixed part of the
  captured packet.
  Default: `false`.
* `--check-xorhop=true|false` - also check XOR applied to the hopping part of
  the captured packet.
  Default: `false`.
* `--check-xordec=true|false` - also check XOR applied after decrypting the
  hopping part.
  Default: `false`.
* `--check-inv-algs=true|false` - also check inverted algorithm variants where
  they exist. Default: `true`.
* `--no-reg-algs=true|false` - disable regular algorithm variants. This is only
  useful when intentionally checking only alternate variants. Default: `false`.

XOR transforms expand the number of input-transform variants the kernels must
check. Enable only the specific XOR locations you need when performance matters.

Selecting a specific learning type usually reduces work, but on some GPUs the
`ALL` path may still be competitive because it avoids some branching patterns.

## Changing CUDA version

### Windows

Update the Visual Studio project files:

* replace `CUDA_PATH_V13_2` with the desired CUDA environment variable
* replace `CUDA 13.2.props` and `CUDA 13.2.targets` with the desired version

### Docker/Linux

Edit `dockerfile`:

```dockerfile
ARG CUDA_MAJOR=13
ARG CUDA_MINOR=2
ARG CUDA_PATCH=0
```
