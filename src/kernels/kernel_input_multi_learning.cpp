#include "kernels/kernel_input_multi_learning.h"

#include "common.h"

#include "device/cuda_array.h"


cudaError_t KeeloqKernelMultiLearningInput::AllocateGPU(size_t totalNumDecryptors, uint8_t numInputs)
{
    auto error = IKeeloqKernelInputBase::AllocateGPU(totalNumDecryptors, numInputs);
    if (error != cudaSuccess)
    {
        return error;
    }

    assert(results == nullptr && "Results data already allocated on GPU");
    if (results == nullptr)
    {
        results = CudaArray<ThreadResult::Multi>::allocate(totalNumDecryptors * numInputs);
    }

    return results != nullptr ? cudaSuccess : cudaGetLastError();
}

void KeeloqKernelMultiLearningInput::FreeGPU()
{

    IKeeloqKernelInputBase::FreeGPU();

    if (results != nullptr)
    {
        results->free();
        results = nullptr;
    }
}

size_t KeeloqKernelMultiLearningInput::BytesAllocated() const
{
    return IKeeloqKernelInputBase::BytesAllocated() + (results ? results->allocated() : 0);
}


BruteforceResult KeeloqKernelMultiLearningInput::getMatch(GetMatchProgressCallback onProgress /*= nullptr*/) const
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
        auto copied_results = CudaArray<ThreadResult::Multi>::read(resultsRam, index, MaxElements);

        for (const auto& result : copied_results)
        {
            if (result.hasMatch())
            {
                auto [Learning, Mod] = KeeloqLearning::DecryptedResults::getByIndex(result.match);

                return BruteforceResult(
                    result.hasMatch(),
                    result.decryptor,
                    result.decrypted.data[result.match],
                    GetInput(result.inputIndex()),
                    result.inputTransform(),
                    Learning,
                    Mod
                );
            }
        }
    }

    return BruteforceResult::Invalid();
}


BruteforceResult KeeloqKernelMultiLearningInput::getResult(size_t index) const
{
    const auto resultsRam = results->host();
    if (index >= resultsRam.num)
    {
        return BruteforceResult::Invalid();
    }

    auto result = results->read(index, 1).front();

    const auto resIndex = result.match;
    auto [learning, mod] = KeeloqLearning::DecryptedResults::getByIndex(resIndex);

    return BruteforceResult(
        result.hasMatch(),
        result.decryptor,
        resIndex < result.decrypted.data.size() ? result.decrypted.data[resIndex] : 0,
        GetInput(result.inputIndex()),
        result.inputTransform(),
        learning,
        mod
    );
}

void KeeloqKernelMultiLearningInput::prepareBatch(const KeeloqLearning::Matrix& learningMatrix, InputTransform inputTransform)
{
    assert(is_valid(inputTransform) && "Invalid input transform");

    assert((GetConfig().type != BruteforceType::XorFix || !!(inputTransform & InputTransform::XorFix)) &&
        "In XorFix bruteforce you should have always XorFix transform enabled");

    learnings = learningMatrix;
    allLearnings = learnings.isAllEnabled();
    activeTransform = inputTransform;

    SetReady(true);
}
