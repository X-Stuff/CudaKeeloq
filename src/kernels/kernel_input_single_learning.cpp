#include "kernels/kernel_input_single_learning.h"

#include "bruteforce/bruteforce_config.h"


cudaError_t KeeloqKernelSingleLearningInput::AllocateGPU(size_t totalNumDecryptors)
{
    auto error = IKeeloqKernelInputBase::AllocateGPU(totalNumDecryptors);
    if (error != cudaSuccess)
    {
        return error;
    }

    assert(results == nullptr && "Results data already allocated on GPU");
    if (results == nullptr)
    {
        results = CudaArray<ThreadResult::Single>::allocate(totalNumDecryptors * IKeeloqKernelInputBase::NumInputs);
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
        auto copied_results = CudaArray<ThreadResult::Single>::read(resultsRam, index, MaxElements);

        // Results are laid out as NumInputs consecutive entries per decryptor, all
        // sharing the same manufacturer key. A decryptor is a full match when those
        // NumInputs entries all matched, so scan for a run of consecutive matching
        // entries that agree on man().
        uint64_t expectedMan = 0;
        int numMatches = 0;

        for (const auto& result : copied_results)
        {
            // A run breaks on a non-matching entry or a change of manufacturer key.
            const bool breaksRun = !result.hasMatch()
                || (expectedMan != 0 && result.decryptor.man() != expectedMan);

            if (breaksRun)
            {
                numMatches = 0;
                expectedMan = 0;
                continue;
            }

            expectedMan = result.decryptor.man();
            ++numMatches;

            if (numMatches == NumInputs)
            {
                return BruteforceResult(
                    true,
                    result.decryptor,
                    result.decrypted,
                    GetInput(result.inputIndex()),
                    result.inputsTransform(),
                    learning,
                    algoType
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
        result.inputsTransform(),
        learning,
        algoType
    );
}

void KeeloqKernelSingleLearningInput::prepareBatch(const KeeloqLearning::Matrix& learningMatrix, InputsTransform inTransform)
{
    if (GetConfig().type == BruteforceType::Xor && !(inTransform & InputsTransform::Xored))
    {
        assert(false && "In Xor bruteforce you should have at least one Xor transform enabled");
        APP_LOG_ERROR(verbosity, "In Xor bruteforce a xor flag for input transform is mandatory, but got %s", InputTransformName(inTransform).c_str());
        return;
    }

    if (!GetConfig().hasSeed() && !!(inTransform & InputsTransform::Xored))
    {
        assert(false && "Xored inputs transform requires a xor value in decryptor (set as seed)");
        APP_LOG_ERROR(verbosity, "%s transform requires a xor value in decryptor (set as seed)", InputTransformName(inTransform).c_str());
        return;
    }

    auto items = learningMatrix.asItems();
    if (items.size() == 0)
    {
        assert(false && "With single learning input there must be only single learning item in matrix!");
        APP_LOG_ERROR(verbosity, "With single learning input there must be only single learning item in matrix, but got empty matrix");
        return;
    }

    if (items.size() > 1)
    {
        assert(false && "With single learning input there must be only single learning item in matrix!");
        APP_LOG_ERROR(verbosity, "With single learning input there must be only single learning item in matrix, but got %zu items", items.size());
        return;
    }

    inputsTransform = inTransform;
    learning = items.front().learning;
    algoType = items.front().algoType;

    SetReady(true);
}
