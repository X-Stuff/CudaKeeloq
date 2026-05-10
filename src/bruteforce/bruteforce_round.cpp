#include "bruteforce/bruteforce_round.h"

#include <cuda_runtime_api.h>

#include "common.h"

#include "algorithm/keeloq/keeloq_encrypted.h"
#include "algorithm/keeloq/keeloq_learning_types.h"
#include "algorithm/keeloq/keeloq_single_result.h"

#include "generators/generator_bruteforce.h"

#include "bruteforce/bruteforce_config.h"
#include "kernels/kernel_result.h"


BruteforceRound::BruteforceRound(const std::vector<EncParcel>& inputs, const BruteforceConfig& config, const CudaConfig& cuda)
    : cudaConfig(cuda)
{
#if NO_INNER_LOOPS
    cudaConfig.substeps = 1;
#endif

    num_decryptors_per_batch = cudaConfig.blocks * cudaConfig.threads * cudaConfig.substeps;

    kernel_inputs.Initialize(config, inputs);
    encrypted_data_num = static_cast<uint8_t>(inputs.size());
}

bool BruteforceRound::readResultsGpu(std::vector<SingleResult>& container) const
{
    assert(inited);
    if (kernel_inputs.results == nullptr)
    {
        printf("Results data is not allocated on GPU. Can't read results.\n");
        return false;
    }

    size_t num_copied = kernel_inputs.results->copy(container);
    return num_copied > 0;
}

bool BruteforceRound::checkResults(const KernelResult& result)
{
    if (result.cudaError != cudaSuccess)
    {
        printf("Kernel fatal error: %s: %s\n Round should be finished!\n", cudaGetErrorName(result.cudaError), cudaGetErrorString(result.cudaError));
        return true;
    }

    if (result.threadsFinished() != cudaConfig.threadsCount())
    {
        printf("Kernel fatal error: Silent kernel crash happened. Expected number of calculations: %u, actual: %u\n",
            cudaConfig.threadsCount(), result.threadsFinished());
        return true;
    }

    if (result.hasMatch())
    {
        printf("Matches count: %d\n", result.hasMatch());
        return true;
    }

    return false;
}

size_t BruteforceRound::getMemSize(bool cpu) const
{
    assert(inited);

    if (cpu)
    {
        return encrypted_data_num * sizeof(EncParcel);
    }
    else
    {
        return kernel_inputs.BytesAllocated();
    }
}

size_t BruteforceRound::numBatches() const
{
    assert(inited);
    if (type() == BruteforceType::Dictionary)
    {
        uint8_t non_align = config().dictSize() % keysPerBatch() == 0 ? 0 : 1;
        return config().dictSize() / keysPerBatch() + non_align;
    }
    else
    {
        uint8_t non_align = config().bruteSize() % keysPerBatch() == 0 ? 0 : 1;
        return config().bruteSize() / keysPerBatch() + non_align;
    }
}

size_t BruteforceRound::keysPerBatch() const
{
    return num_decryptors_per_batch;
}

size_t BruteforceRound::resultsPerBatch() const
{
    return encrypted_data_num * num_decryptors_per_batch;
}

void BruteforceRound::init()
{
    if (!inited)
    {
        alloc();

        inited = true;
    }
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
        "----------------------------------------\n"
        "Config: %s\n"
        "----------------------------------------",
        cudaConfig.blocks, cudaConfig.threads, cudaConfig.substeps, (getMemSize(false) / static_cast<float>(1024 * 1024 * 1024)),
        encrypted_data_num, config().mutationsToString().c_str(),
        resultsPerBatch(), keysPerBatch(), config().toString().c_str());
}


bool BruteforceRound::prepareInputs(uint64_t batchIdx)
{
    if (type() != BruteforceType::Dictionary)
    {
        if (batchIdx != 0)
        {
            // Make last decryptor from previous batch as first for this batch
            kernel_inputs.NextDecryptor();
        }

        // Generate decryptors (if available)
        auto cudaError = GeneratorBruteforce::PrepareDecryptors(kernel_inputs, cudaConfig);
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
        kernel_inputs.WriteDecryptors(config().decryptors, batchIdx * keysPerBatch(), keysPerBatch());
    }

    return true;
}

void BruteforceRound::alloc()
{
    //
    assert(kernel_inputs.decryptors == nullptr	&& "Decryptors data already allocated on GPU");
    assert(kernel_inputs.results == nullptr		&& "Results data already allocated on GPU");


    // ALLOCATE ON GPU
    if (kernel_inputs.decryptors == nullptr)
    {
        kernel_inputs.decryptors = CudaArray<Decryptor>::allocate(num_decryptors_per_batch);
    }

    if (kernel_inputs.results == nullptr)
    {
        kernel_inputs.results = CudaArray<SingleResult>::allocate(resultsPerBatch());
    }
}

void BruteforceRound::free()
{
    if (kernel_inputs.decryptors != nullptr)
    {
        kernel_inputs.decryptors->free();
        kernel_inputs.decryptors = nullptr;
    }

    if (kernel_inputs.results != nullptr)
    {
        kernel_inputs.results->free();
        kernel_inputs.results = nullptr;
    }
}
