#pragma once

#include <string>
#include <vector>

#include "common.h"

#include "device/cuda_config.h"

#include "kernels/kernel_result.h"
#include "kernels/kernel_input_multi_learning.h"

#include "algorithm/keeloq/keeloq_encrypted.h"
#include "algorithm/keeloq/keeloq_learning_types.h"
#include "algorithm/keeloq/keeloq_single_result.h"

#include "bruteforce/bruteforce_config.h"


/**
 * A single attack "round" — the full sweep of one BruteforceConfig.
 *
 * A round is composed of batches; each batch runs `B` CUDA blocks × `T` threads,
 * each thread checks one or more decryptors, and each check evaluates one or
 * more keeloq learning types.
 *
 * Total batches in a round ≈ (keys to check) / (B * T * I).
 */
struct BruteforceRound
{
    BruteforceRound(const std::vector<EncParcel>& data, const BruteforceConfig& gen, const CudaConfig& config);

    ~BruteforceRound()
    {
        free();
    }

public:
    /** One-shot allocation of GPU buffers; must be called before running batches. */
    void init();

    /** Reads GPU result buffer into the supplied container; returns true if any data was copied. */
    bool readResultsGpu(std::vector<SingleResult>& container) const;

    /** Inspects a kernel result and returns true if the round should stop (match or fatal error). */
    bool checkResults(const KernelResult& result, AppVerbosity verbosity = AppVerbosity::Debug);

    /** Allocated memory (bytes) for CPU or GPU buffers. */
    size_t getMemSize(bool cpu = false) const;

    /** Number of batches required to cover the configured key space. */
    size_t numBatches() const;

    /** Result slots written per batch (num inputs × keys per batch). */
    size_t resultsPerBatch() const;

    /** Decryptor count processed per batch. */
    size_t keysPerBatch() const;

    /** Human-readable summary of the round's CUDA config, memory, and attack parameters. */
    std::string toString() const;

public:
    /** Underlying bruteforce configuration (requires init()). */
    inline const BruteforceConfig& config() const { assert(inited); return kernel_inputs.GetConfig(); }

    /** Bruteforce attack type (requires init()). */
    inline BruteforceType::Type type() const { assert(inited); return config().type; }

    /** Mutable kernel inputs (requires init()). */
    inline KeeloqKernelMultiLearningInput& inputs() { assert(inited); return kernel_inputs; }

    /** Const kernel inputs (requires init()). */
    inline const KeeloqKernelMultiLearningInput& inputs() const { assert(inited); return kernel_inputs; }

    /** Launches decryptors generator kernels and returns true if succeeded. */
    bool prepareInputs(uint64_t batchIdx);

private:

    void alloc();

    void free();

private:

    bool inited = false;

    // NumBlocks * NumThreads * [NumIterations] (num iteration is 1 if NO_INNER_LOOPS is defined, by default)
    uint32_t num_decryptors_per_batch = 0;

    //
    KeeloqKernelMultiLearningInput kernel_inputs;

    uint8_t encrypted_data_num = 0;

    CudaConfig cudaConfig;
};
