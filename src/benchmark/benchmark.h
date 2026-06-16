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
    int run(const std::vector<EncParcel>& inputs, const BruteforceConfig& benchmarkConfig, uint32_t CudaBlocks, uint16_t CudaThreads);

    /** Sweep over several block/thread configurations for a single bruteforce config. */
    void run(const std::vector<EncParcel>& inputs, const KeeloqLearning::Matrix& learningMatrix, const BruteforceConfig& benchmarkConfig);

    /** "Real" captures benchmark: runs known keeloq targets (FAAC SLH, DH, Sommer) and verifies matches. */
    void benchmarkReal(bool useSingleLearningKernels);

    /** Run benchmark for each learning with simple and alphabet configs */
    void benchmarkEveryLearningAlone(uint32_t TargetCalculationsNumber);

    /** Run benchmark for all learnings with simple +1 config */
    void benchmarkEveryLearningAtOnce(uint32_t TargetCalculationsNumber);

    /** Run benchmark for seed-only attack */
    void benchmarkSeedAttack(uint32_t TargetCalculationsNumber);

    /** Run benchmark for normal-only attack */
    void benchmarkNormalAttack(uint32_t TargetCalculationsNumber);

    /** Run benchmark for XOR-based attack */
    void benchmarkXoredAttack(uint32_t TargetCalculationsNumber);

    /** Runs the full benchmark suite, including the real captures and all configured sweeps. */
    void all(const CommandLineArgs& args);
}
