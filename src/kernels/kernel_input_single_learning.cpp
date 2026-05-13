#include "kernels/kernel_input_single_learning.h"

#include "bruteforce/bruteforce_config.h"


cudaError_t KeeloqKernelSingleLearningInput::AllocateGPU(size_t totalNumDecryptors, uint8_t numInputs)
{
    auto error = IKeeloqKernelInputBase::AllocateGPU(totalNumDecryptors, numInputs);
    if (error != cudaSuccess)
    {
        return error;
    }

    assert(results == nullptr && "Results data already allocated on GPU");
    if (results == nullptr)
    {
        results = CudaArray<SingleLearningResult>::allocate(totalNumDecryptors * numInputs);
    }

    return results != nullptr ? cudaSuccess : cudaGetLastError();
}


void KeeloqKernelSingleLearningInput::FreeGPU()
{
    IKeeloqKernelInputBase::FreeGPU();

    if (results != nullptr)
    {
        results->free();
        results = nullptr;
    }
}

size_t KeeloqKernelSingleLearningInput::BytesAllocated() const
{
    return TKeeloqKernelInputBase::BytesAllocated() + (results ? results->allocated() : 0);
}


BruteforceResult KeeloqKernelSingleLearningInput::getMatch(GetMatchProgressCallback onProgress/*= nullptr*/) const
{
    constexpr auto MaxElements = 1024 * 1024;

    const auto resultsRam = results->host();
    const auto numIterations = resultsRam.num / MaxElements;

    for (size_t index = 0, count = 0; index < resultsRam.num; index += MaxElements, ++count)
    {
        if (onProgress)
        {
            onProgress(count, numIterations);
        }

        // Read decryptors in batches, just to save RAM on host, using static method since we already copied object to host
        auto copied_results = CudaArray<SingleLearningResult>::read(resultsRam, index, MaxElements);

        for (const auto& result : copied_results)
        {
            if (result.hasMatch())
            {
                return BruteforceResult(
                    true,
                    result.decryptor,
                    result.decrypted,
                    GetInput(result.inputIndex()),
                    result.inputsMutation(),
                    learning,
                    algorithModifier
                );
            }
        }
    }

    return BruteforceResult::Invalid();
}


BruteforceResult KeeloqKernelSingleLearningInput::getResult(size_t index) const
{
    const auto resultsRam = results->host();
    if (index >= resultsRam.num)
    {
        return BruteforceResult::Invalid();
    }

    auto result = results->read(index, 1).front();

    return BruteforceResult(
        result.hasMatch(),
        result.decryptor,
        result.decrypted,
        GetInput(result.inputIndex()),
        result.inputsMutation(),
        learning,
        algorithModifier
    );
}

void KeeloqKernelSingleLearningInput::BruteforcePrepare(InputsMutation mutations, KeeloqLearning::LearningType learningType, KeeloqLearning::Modifier::Algo algorithModifier)
{
    inputsMutation = mutations;
    learning = learningType;
    algorithModifier = algorithModifier;

    SetReady(true);
}