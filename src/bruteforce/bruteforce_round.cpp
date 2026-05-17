#include "bruteforce/bruteforce_round.h"

#include <cuda_runtime_api.h>

#include "common.h"

#include "algorithm/keeloq/keeloq_kernel.h"
#include "algorithm/keeloq/keeloq_encrypted.h"
#include "algorithm/keeloq/keeloq_learning_types.h"
#include "algorithm/keeloq/keeloq_thread_result.h"

#include "generators/generator_bruteforce.h"

#include "bruteforce/bruteforce_config.h"

#include "kernels/kernel_result.h"
#include "kernels/kernel_input_multi_learning.h"
#include "kernels/kernel_input_single_learning.h"


BruteforceRound::BruteforceRound(const std::vector<EncParcel>& inputs, const BruteforceConfig& config, const CudaConfig& cuda)
    : cudaConfig(cuda), inputsNum(static_cast<uint8_t>(inputs.size()))
{
#if NO_INNER_LOOPS
    cudaConfig.substeps = 1;
#endif

    num_decryptors_per_batch = cudaConfig.blocks * cudaConfig.threads * cudaConfig.substeps;

    kernel_inputs = config.useSingleLearningKernels ?
        IKeeloqKernelInputBase::Create<KeeloqKernelSingleLearningInput>() :
        IKeeloqKernelInputBase::Create<KeeloqKernelMultiLearningInput>();

    kernel_inputs->Initialize(config, inputs);
}

bool BruteforceRound::checkResults(const KernelResult& result, AppVerbosity verbosity)
{
    if (result.cudaError != cudaSuccess)
    {
        APP_LOG_ERROR(verbosity, "Kernel fatal error: %s: %s\n Round should be finished!\n", cudaGetErrorName(result.cudaError), cudaGetErrorString(result.cudaError));
        return true;
    }

    if (result.threadsFinished() != cudaConfig.threadsCount())
    {
        APP_LOG_ERROR(verbosity, "Kernel fatal error: Silent kernel crash happened. Expected number of calculations: %u, actual: %u\n",
            cudaConfig.threadsCount(), result.threadsFinished());
        return true;
    }

    if (result.hasMatch())
    {
        APP_LOG_INFO(verbosity, "Matches count: %d\n", result.hasMatch());
        return true;
    }

    return false;
}

size_t BruteforceRound::getMemSize(bool cpu) const
{
    assert(inited);

    if (cpu)
    {
        return inputsNum * sizeof(EncParcel);
    }
    else
    {
        return kernel_inputs->BytesAllocated();
    }
}

size_t BruteforceRound::numBatches() const
{
    assert(inited);
    if (type() == BruteforceType::Dictionary)
    {
        const uint8_t non_align = config().dictSize() % decryptorsPerBatch() == 0 ? 0 : 1;
        return config().dictSize() / decryptorsPerBatch() + non_align;
    }
    else
    {
        const uint8_t non_align = config().bruteSize() % decryptorsPerBatch() == 0 ? 0 : 1;
        return config().bruteSize() / decryptorsPerBatch() + non_align;
    }
}

size_t BruteforceRound::decryptorsPerBatch() const
{
    return num_decryptors_per_batch;
}

size_t BruteforceRound::resultsPerBatch() const
{
    return decryptorsPerBatch() * inputsNum;
}

void BruteforceRound::init()
{
    if (!inited)
    {
        alloc();

        inited = true;
    }
}

KernelResult BruteforceRound::update(const KeeloqLearning::Matrix& learningMatrix, InputsMutation inputsMutation)
{
    if (!kernel_inputs)
    {
        assert(kernel_inputs && "Inputs were not constructed!");
        return KernelResult::NotStarted();
    }

    kernel_inputs->prepareBatch(learningMatrix, inputsMutation);
    return keeloq::kernels::cuda_brute(*this, cudaConfig);
}

std::string BruteforceRound::toString() const
{
    assert(inited);

    return str::format<std::string>(
        "----------------------------------------\n"
        "CUDA:\n"
        "\t- Blocks:%u\n"
        "\t- Threads:%u\n"
        "\t- Substeps:%u\n"
        "\t- Allocated GPU memory: %-2.1f GB\n"
        "----------------------------------------\n"
        "Inputs:\n"
        "\tCount: %u\n"
        "\tMutations:   %s\n"
        "----------------------------------------\n"
        "Results per batch:   %8zd\n"
        "Decryptors per batch:%8zd\n"
        "Kernel Inputs:       %8s\n"
        "----------------------------------------\n"
        "Config: %s\n"
        "----------------------------------------",
        cudaConfig.blocks, cudaConfig.threads, cudaConfig.substeps, (getMemSize(false) / static_cast<float>(1024 * 1024 * 1024)),
        inputsNum, config().mutationsToString().c_str(),
        resultsPerBatch(), decryptorsPerBatch(), kernel_inputs ? ::toString(kernel_inputs->type()).data() : "NULL",
        config().toString().c_str());
}


bool BruteforceRound::generateDecryptors(uint64_t batchIdx)
{
    if (type() != BruteforceType::Dictionary)
    {
        if (batchIdx != 0)
        {
            // Make last decryptor from previous batch as first for this batch
            inputs().NextDecryptor();
        }

        // Generate decryptors (if available)
        auto cudaError = GeneratorBruteforce::PrepareDecryptors(inputs(), cudaConfig);
        if (cudaError != cudaSuccess)
        {
            printf("Error: Key generation resulted with error: %s: %s\n", cudaGetErrorName(cudaError), cudaGetErrorString(cudaError));
            assert(false);
            return false;
        }
    }
    else
    {
        // Write next batch of keys from dictionary
        inputs().WriteDecryptors(config().dictDecryptors, batchIdx * decryptorsPerBatch(), decryptorsPerBatch());
    }

    return true;
}

void BruteforceRound::alloc()
{
    // ALLOCATE ON GPU
    inputs().AllocateGPU(num_decryptors_per_batch, inputsNum);
}

void BruteforceRound::free()
{
    inputs().FreeGPU();
}
