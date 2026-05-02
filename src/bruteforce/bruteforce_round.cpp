#include "bruteforce_round.h"

#include "common.h"
#include "bruteforce_config.h"
#include "kernels/kernel_result.h"
#include "algorithm/keeloq/keeloq_encrypted.h"
#include "algorithm/keeloq/keeloq_learning_types.h"
#include "algorithm/keeloq/keeloq_single_result.h"

#include <cuda_runtime_api.h>


BruteforceRound::BruteforceRound(const std::vector<EncParcel>& inputs, const BruteforceConfig& config, const KeeloqLearning::Matrix& learning_matrix, const CudaConfig& cuda)
    : cudaConfig(cuda)
{
#if NO_INNER_LOOPS
    cudaConfig.substeps = 1;
#endif

    num_decryptors_per_batch = cudaConfig.blocks * cudaConfig.threads * cudaConfig.substeps;

    kernel_inputs.Initialize(config, inputs, learning_matrix);
    encrypted_data_num = static_cast<uint8_t>(inputs.size());
}

bool BruteforceRound::read_results_gpu(std::vector<SingleResult>& container) const
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

bool BruteforceRound::check_results(const KernelResult& result)
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

size_t BruteforceRound::get_mem_size(bool cpu) const
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

size_t BruteforceRound::num_batches() const
{
    assert(inited);
    if (Type() == BruteforceType::Dictionary)
    {
        uint8_t non_align = Config().dict_size() % keys_per_batch() == 0 ? 0 : 1;
        return Config().dict_size() / keys_per_batch() + non_align;
    }
    else
    {
        uint8_t non_align = Config().brute_size() % keys_per_batch() == 0 ? 0 : 1;
        return Config().brute_size() / keys_per_batch() + non_align;
    }
}

size_t BruteforceRound::keys_per_batch() const
{
    return num_decryptors_per_batch;
}

size_t BruteforceRound::results_per_batch() const
{
    return encrypted_data_num * num_decryptors_per_batch;
}

void BruteforceRound::Init()
{
    if (!inited)
    {
        alloc();

        inited = true;
    }
}

std::string BruteforceRound::to_string() const
{
    assert(inited);

    return str::format<std::string>("Setup:\n"
        "\tCUDA: Blocks:%u Threads:%u Substeps:%u\n"
        "\tCUDA: Used memory: %.1f GB\n"
        "\tInputs num: %u\n"
        "\tResults per batch: %zd\n"
        "\tDecryptors per batch:%zd\n"
        "\tConfig: %s\n"
        "\tLearning: %s\n",
        cudaConfig.blocks, cudaConfig.threads, cudaConfig.substeps, (get_mem_size(false) / static_cast<float>(1024 * 1024 * 1024)),
        encrypted_data_num, results_per_batch(), keys_per_batch(), Config().toString().c_str(),
        kernel_inputs.GetLearningMatrix().to_string(&Config()).c_str());
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
        kernel_inputs.results = CudaArray<SingleResult>::allocate(results_per_batch());
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
