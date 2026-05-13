#pragma once

#include <cstring>

#include "common.h"

#include "device/cuda_array.h"
#include "device/cuda_object.h"

#include "kernels/kernel_input_base.h"

#include "algorithm/keeloq/keeloq_decryptor.h"
#include "algorithm/keeloq/keeloq_encrypted.h"
#include "algorithm/keeloq/keeloq_learning_types.h"
#include "algorithm/keeloq/keeloq_single_result.h"
#include "algorithm/keeloq/keeloq_learning_result.h"

#include "bruteforce/bruteforce_config.h"

/**
 * Host/device-shared input bundle for the main keeloq bruteforce kernel.
 * Owns the per-run decryptor batch, result buffer, learning matrix, and bruteforce config.
 */
struct KeeloqKernelMultiLearningInput : public TKeeloqKernelInputBase<KeeloqKernelMultiLearningInput>
{
public:
    KeeloqKernelMultiLearningInput() : TKeeloqKernelInputBase<KeeloqKernelMultiLearningInput>(this)
    {
    }

    KeeloqKernelMultiLearningInput(KeeloqKernelMultiLearningInput&& other) noexcept
        : TKeeloqKernelInputBase<KeeloqKernelMultiLearningInput>(std::move(other), this)
    {
        results = other.results;
        learnings = other.learnings;
        allLearnings = other.allLearnings;
        mutationsMask = other.mutationsMask;

        other.results = nullptr;
    }

    KeeloqKernelMultiLearningInput& operator=(KeeloqKernelMultiLearningInput&& other) = delete;
    KeeloqKernelMultiLearningInput& operator=(const KeeloqKernelMultiLearningInput& other) = delete;

public:
    virtual __host__ cudaError_t AllocateGPU(size_t totalNumDecryptors, uint8_t numInputs) final override;

    /** Release allocated GPU memory for results and decryptors */
    virtual __host__ void FreeGPU() final override;

    /** Bytes allocated by the decryptor and result buffers (for metrics). */
    virtual __host__ size_t BytesAllocated() const override;

    /** Returns bruteforce match found in results if any */
    virtual __host__ BruteforceResult getMatch(GetMatchProgressCallback = nullptr) const final override;

    /** Get specific result by index */
    virtual __host__ BruteforceResult getResult(size_t index) const final override;

public:
    /** Learning-type selection matrix for the current run. */
    __device__ __host__ __inline__ const KeeloqLearning::Matrix& GetLearningMatrix() const { return learnings; }

    /** Fast path indicator: true when every learning entry is enabled. */
    __device__ __host__ __inline__ bool AllLearningsEnabled() const { return allLearnings; }

    /** Active input mutation for the next kernel launch. */
    __device__ __host__ __inline__ InputsMutation InputsMutationMask() const { return mutationsMask; }

    /** Each time before bruteforce called, this method should be called */
    __host__ void BruteforcePrepare(const KeeloqLearning::Matrix& inLearnings, InputsMutation mutations);

public:
    // Single-run results accessible from GPU
    CudaArray<SingleResult>* results = nullptr;

private:
    // Which type of learning use for decryption
    KeeloqLearning::Matrix learnings;

    // optimizations. Just a bool field that could be accessed from GPU
    bool allLearnings = false;

    // Inputs mutation mask
    InputsMutation mutationsMask = InputsMutation::None;
};
