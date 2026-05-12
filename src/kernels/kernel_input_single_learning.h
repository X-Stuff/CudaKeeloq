#pragma once

#include "common.h"

#include "device/cuda_array.h"
#include "device/cuda_object.h"

#include "kernels/kernel_input_base.h"

#include "algorithm/keeloq/keeloq_learning_result.h"
#include "algorithm/keeloq/keeloq_learning_types.h"

#include "bruteforce/bruteforce_config.h"


/**
 *  Experimental: flattened version of KeeloqKernelInput for testing whether it can improve GPU performance by better memory access patterns.
 * In this mode CUDA kernel computes only one variation of learning/mutation/algomod, but shares produced decryptors
 */
struct KeeloqKernelSingleLearningInput : public TKeeloqKernelInputBase<KeeloqKernelSingleLearningInput>
{
public:
    KeeloqKernelSingleLearningInput() : TKeeloqKernelInputBase<KeeloqKernelSingleLearningInput>(this)
    {
    }

    KeeloqKernelSingleLearningInput(KeeloqKernelSingleLearningInput&& other) noexcept : TKeeloqKernelInputBase<KeeloqKernelSingleLearningInput>(this)
    {
        inputsCount = other.inputsCount;
        decryptors = other.decryptors;
        results = other.results;
        inputsMutation = other.inputsMutation;
        learning = other.learning;
        algorithModifier = other.algorithModifier;
    }

    virtual ~KeeloqKernelSingleLearningInput() override = default;

    KeeloqKernelSingleLearningInput& operator=(KeeloqKernelSingleLearningInput&& other) = delete;
    KeeloqKernelSingleLearningInput& operator=(const KeeloqKernelSingleLearningInput& other) = delete;
public:
    /** Bytes allocated by the decryptor and result buffers (for metrics). */
    virtual __host__ size_t BytesAllocated() const override;

public:
    /** Each time before bruteforce called, this method should be called */
    void BruteforcePrepare(InputsMutation mutations, KeeloqLearning::LearningType learningType, KeeloqLearning::Modifier::Algo algorithModifier);

public:
    template<uint8_t InputIndex, uint8_t NumInputs>
    __device__ __forceinline__ SingleLearningResult& Result(size_t decryptorIndex)
    {
        return (*results)[decryptorIndex * NumInputs + InputIndex];
    }

    __device__ __forceinline__ const Decryptor& GetDecryptor(size_t decryptorIndex)
    {
        return (*decryptors)[decryptorIndex];
    }

public:
    // Per-variation results, size is number of decryptors * inputsNum
    CudaArray<SingleLearningResult>* results = nullptr;

    // How inputs should be mutated in Kernel
    InputsMutation inputsMutation = InputsMutation::None;

    // What type of leaning should Kernel use for decryption
    KeeloqLearning::LearningType learning = KeeloqLearning::LearningType::Simple;

    // Is there any algorithm modifier (like inversed Dec/Enc) that should be applied to inputs in Kernel
    KeeloqLearning::Modifier::Algo algorithModifier = KeeloqLearning::Modifier::Algo::Normal;
};