#pragma once

#include <vector>

#include "common.h"

#include "host/command_line_args.h"

struct BruteforceConfig;

/**
 * Bruteforce benchmark harness.
 *
 * Sweeps CUDA launch configurations for a given config / learning matrix and
 * reports throughput. Ships as part of the main binary (--benchmark).
 */
namespace benchmark
{
    /**
     *  Run a single configuration at specific block/thread counts.
     * Returns avg. speed (bigger-better, negative - failure, 0 - cancel request)
     */
    int run(const std::vector<EncParcel>& inputs, const KeeloqLearning::Matrix& learningMatrix,
        const BruteforceConfig& benchmarkConfig, uint32_t CudaBlocks, uint16_t CudaThreads, uint32_t TargetKeysCount);

    /** Sweep over several block/thread configurations for a single bruteforce config. */
    void run(const std::vector<EncParcel>& inputs, const KeeloqLearning::Matrix& learningMatrix,
        const BruteforceConfig& benchmarkConfig, uint32_t TargetKeysCount);

    /** "Real" captures benchmark: runs known keeloq targets (DH, Sommer) and verifies matches. */
    void real();

    /** Runs the full benchmark suite, including the real captures and all configured sweeps. */
    void all(const CommandLineArgs& args);
}
