#pragma once

#include <string>
#include <vector>
#include <memory>

#include "common.h"

#include "device/cuda_config.h"

#include "kernels/kernel_result.h"
#include "kernels/kernel_input_multi_learning.h"

#include "algorithm/keeloq/keeloq_encrypted.h"
#include "algorithm/keeloq/keeloq_learning_types.h"
#include "algorithm/keeloq/keeloq_thread_result.h"

#include "bruteforce/bruteforce_config.h"


/**
 * A single attack "round" — the full sweep of one BruteforceConfig.
 *
 * A round is composed of batches; each batch runs `B` CUDA blocks x `T` threads,
 * each thread checks one or more decryptors, and each check evaluates one or
 * more keeloq learning types.
 *
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

    /** Launches decryptors generator kernels and returns true if succeeded. */
    bool generateDecryptors(uint64_t batchIdx);

    /** Launch bruteforce kernel and get result */
    KernelResult update(const KeeloqLearning::Matrix& learningMatrix, InputsTransform inTransform);

    /** Inspects a kernel result and returns true if the round should stop (match or fatal error). */
    bool checkResults(const KernelResult& result, AppVerbosity verbosity = AppVerbosity::Debug);

public:
    /** Allocated memory (bytes) for CPU or GPU buffers. */
    size_t getMemSize(bool cpu = false) const;

    /** Number of batches required to cover the configured key space. */
    size_t numBatches() const;

    /** Result slots written per batch (num inputs × keys per batch). */
    size_t resultsPerBatch() const;

    /** Decryptor count processed per batch. */
    size_t decryptorsPerBatch() const;

    /** Human-readable summary of the round's CUDA config, memory, and attack parameters. */
    std::string toString() const;

public:
    /** Underlying bruteforce configuration (requires init()). */
    inline const BruteforceConfig& config() const { assert(inited); return inputs().GetConfig(); }

    /** Bruteforce attack type (requires init()). */
    inline BruteforceType::Type type() const { assert(inited); return config().type; }

    /** Mutable kernel inputs (requires init()). */
    inline IKeeloqKernelInputBase& inputs() { assert(kernel_inputs); return *kernel_inputs; }

    /** Const kernel inputs (requires init()). */
    inline const IKeeloqKernelInputBase& inputs() const { assert(kernel_inputs); return *kernel_inputs; }

    /** Returns true if the current inputs are for single learning brute mode. */
    inline bool isSingleLearningInputs() const { assert(kernel_inputs); return kernel_inputs->type() == IKeeloqKernelInputBase::Type::Single; }

private:

    void alloc();

    void free();

private:

    CudaConfig cudaConfig;

    // Cached number of inputs was used to create this round
    const uint8_t inputsNum = 0;

    //
    std::unique_ptr<IKeeloqKernelInputBase> kernel_inputs;

    // NumBlocks * NumThreads * [NumIterations] (num iteration is 1 if NO_INNER_LOOPS is defined, by default)
    uint32_t num_decryptors_per_batch = 0;

    bool inited = false;
};
