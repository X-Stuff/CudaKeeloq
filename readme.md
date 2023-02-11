## Intro
This is a CUDA accelerated simple bruteforcer of [KeeLoq](https://en.wikipedia.org/wiki/KeeLoq) algorithm.

## Disclaimer

> 64-bit keeloq key is 18,446,744,073,709,551,615 possible combinations
> EVEN! if your GPU will be able to calculate 1 billion keys in a second
> You will need 18446744073709551615 / 1000000000 / 3600 / 24 / 365 = 584 YEARS! to brute a single key.
> My laptop 3080Ti can do only 230 MKeys/s
> So it's practically impossible to use this application "as is" in real life attack.

## Capabilities

* **Simple** (+1) bruteforce
  > Regular straightforward bruteforce where keys just incremented +1
* **Filtered** (+1) bruteforce
  > Some basics filters for simple bruteforce, like "keys with more that 6 subsequent zeros". Filters may apply to `Include` and/or `Exclude` rules. e.g. you can exclude keys with all ASCII symbols. You may also use `Include` filter but its performance is incredibly bad, so I don't recommend this attack type at all.
* **Alphabet** attack
  > You may specify exact set of bytes which should be used for generating keys during bruteforce. For example you may want to use only numbers, or only ascii symbols.
* **Pattern** attack
  > This is like extended alphabet. You may specify individual set of bytes for each byte in 64bit key. For example you may want first byte be any value, but others be only numbers.

## Build

### Windows
#### Requirements

* CUDA Toolkit v12.0.0
  - nvcc
  - cuda runtime
  - visual studio extension

* Microsoft Visual Studio 2022

#### Compiling
 Just open `.vcxproj` file and build it

### Linux
#### Requirements
* docker

#### Compiling
```
$ ./build.sh
```
This will create a container `cudakeeloq:local` with compiled app

Run the bruteforcer
```
$ ./run.sh
```
> NOTE: You may need to have CUDA docker extension installed in order to have `--gpus` command line argument works.

## Launching

### Requirements
* NVidia GPU (1GB+ RAM)
* RAM 1GB+
* **(Linux only)** installed CUDA docker extension `nvidia-container-toolkit`

### Command line

 * You may specify a bruteforce starting key and count how much keys should be checked.
 * You may specify several attack modes. First check first alphabet, then other.
   > NOTE: At the moment you cannot specify start and count for each mode, only for whole launch
 * Launch with `--test` argument to run basic tests (using outside `DEBUG` mode (not available in Github releases) might be useless)
 * Launch with `--benchmark` arguments will run benchmark tests to show you performance per each CUDA configuration (threads, blocks)

### Examples

```
./CudaKeeloq --inputs xx,yy,zz --mode=1 --start=0x9876543210 --count=1000000
```
Simple bruteforce of 1 million keys starting from `0x9876543210`

```
./CudaKeeloq --inputs xx,yy,zz --mode=3 --learning-type=0 --alphabet=examples/alphabet.bin,10:20:30:AA:BB:CC:DD:EE:FF:02:33
```
Alphabet attacks with 2 dictionaries, one is in file `examples/alphabet.bin` second is provided via command line `10:20:30:AA:BB:CC:DD:EE:FF:02:33`

### Inputs

* `--inputs=[i1, i2, i3]` - inputs are captured "over the air" bytes, you need to provide 3 in format of hexadecimal number: `0x1122334455667788`

### CUDA Setup

* `--cuda-blocks=N` - `N` is a number of thread blocks for each bruteforce attack round
* `--cuda-threads=N` - `N` is a number of CUDA threads for each block (above). If `0` (default) - the maximum from device caps will be set.

Overall number of threads is multiplication of `cuda-blocks` and `cuda-threads`.

Keep in mind that the more overall threads you will have - the more memory they will consume. (e.g. `8196` and `1024` consumes approx. 2.5 GB RAM (and GPU RAM))

### Attack modes

Each bruteforce run must define a mode.
Several modes can be specified at the same time like `--mode=0,1,3,4` but you should provide all expected arguments for each mode.

 * `--mode=0` - Dictionary. Expects (one or both):
    - `--word-dict=[f1, w1, ...]` - `f1` - text dictionary file(s) with hexadecimal values, `w1` hexadecimal word itself
    - `--bin-dict=[f1, f2]` - binary file(s) each 8 bytes of which will be used as dictionary word.
    Supports `--bin-dict-mode=M` where `M` = { `0`, `1`, `2`}. `0` - as is (big endian). `1` - reverse (little endian), `2` - both.
 * `--mode=1` - Simple bruteforce mode.
    - `--start` defines the first key from which bruteforce begins.
    - `--count` how much keys to check.
 * `--mode=2` - Filtered bruteforce mode. Same as simple, but with filters:
    - `--exclude-filter=V` where `V` is number representing filers flags combinations. (see: [bruteforce_filters.h](src/bruteforce/bruteforce_filters.h))
    - `--include-filter=V` (same as above)
 * `--mode=3` - Alphabet mode. Expects:
    - `--alphabet=[f1, a1, ...]` - where `f1` is a binary file contents of which will be interpreted as allowed bytes in key during bruteforce. where `a1` is `:` separated hex string of bytes in alphabet (like: `AA:BB:01`)
    - Also allowed to use `--start` and `--count` arguments, with same meaning
 * `--mode=4` - Pattern mode. Expects:
    - `--pattern=[f1, p1, ...]` - where `f1` is a text file with hexadecimal pattern, and `p1` is `:` separated hexadecimal pattern like `*:aa:00-33:0xF3|0x11:AL0 ...`.
        - `*` - means any byte
        - `0xNN-0xMM` - means range from `0xNN` to `0xMM`
        - `A|B|C` - means `A` or `B` or `C`
        - `AL[0..N]` - means alphabet from inputs (must be specified with `--alphabet=`)
    - `--alphabet=` - if pattern has `AL` - alphabet must be specified.

### Learning types

By default the app will try to brute all known learning keys (16 or 12 depending if seed is specified), however if you know exact learning type, you might increase you bruteforce time x12-16 times. You may also specify several learning types simultaneously `--learning-type=0,5,7`.
> NOTE: At some point using `ALL` learning types might be faster than explicitly specified due to GPU branching problem.

Each learning type has its `_REV` modification. That's mean it will use byte-reversed key for decryption. See more [keeloq_learning_types.h](src/algorithm/keeloq/keeloq_learning_types.h)

Here and below `=x[y]` - `x` value for normal mode, `y` for reversed.

* `--learning-type=0[1]` - Simple learning
* `--learning-type=2[3]` - Normal learning
* `--learning-type=4[5]` - Secure learning (requires seed)
* `--learning-type=6[7]` - XOR learning
* `--learning-type=8[9]` - FAAC learning (requires seed)
* `--learning-type=10[11]` - UNKNOWN TYPE1
* `--learning-type=12[13]` - UNKNOWN TYPE2
* `--learning-type=14[15]` - UNKNOWN TYPE3

### Miscellaneous

 * `--first-match` - if `true` (default) will stop all bruteforce on first match
 * `--test` - launches internal debug tests
 * `--benchmark` - launches CUDA setup benchmark
 * `--help`, `-h` - prints help


 ## Performance

 For my laptop's GPU ( 3080Ti ) the best results with `8196` CUDA Blocks and maximum CUDA threads (from device info - `1024`) - it gives me approx. `15` MKeys/s for `ALL` learning types and `230` MKeys for `Simple`.