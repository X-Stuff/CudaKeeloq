#include "bruteforce_round.h"

#include "common.h"
#include "bruteforce_config.h"
#include "kernels/kernel_result.h"
#include "algorithm/keeloq/keeloq_encrypted.h"
#include "algorithm/keeloq/keeloq_learning_types.h"
#include "algorithm/keeloq/keeloq_single_result.h"

#include <cuda_runtime_api.h>


BruteforceRound::BruteforceRound(const std::vector<EncParcel>& enc, const BruteforceConfig& config, const KeeloqLearning::Matrix& learning_matrix,
    uint32_t blocks, uint32_t threads, uint32_t iterations)
    : encrypted_data(enc)
{
#if NO_INNER_LOOPS
    iterations = 1;
#endif

    CUDASetup[0] = blocks;
    CUDASetup[1] = threads;
    CUDASetup[2] = iterations;

    num_decryptors_per_batch = iterations * threads * blocks;

    kernel_inputs.Initialize(config, learning_matrix);
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
    if (result.error < 0)
    {
        printf("Kernel fatal error: %d\n Round should be finished!\n", result.error);
        return true;
    }
    else if (result.error != 0)
    {
        printf("CUDA calculations num errors: %d\n", result.error);
    }

    if (result.value > 0)
    {
        printf("Matches count: %d\n", result.value);

        std::vector<SingleResult> all_results;
        if (!read_results_gpu(all_results))
        {
            printf("Failed to read results from GPU. Round should be finished!\n");
            return true;
        }

        for (const auto& result : all_results)
        {
            if (result.match == KeeloqLearning::NoMatch)
            {
                continue;
            }

            result.print();
        }

        return true;
    }

    return false;
}

size_t BruteforceRound::get_mem_size(bool cpu) const
{
    assert(inited);

    if (cpu)
    {
        return encrypted_data.size() * sizeof(EncParcel);
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
    return encrypted_data.size() * num_decryptors_per_batch;
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
        "\tCUDA: Blocks:%u Threads:%u Iterations:%u\n"
        "\tCUDA: Used memory:%zd\n"
        "\tEncrypted data size:%zd\n"
        "\tResults per batch:%zd\n"
        "\tDecryptors per batch:%zd\n"
        "\tConfig: %s"
        "\tLearning Matrix:%s\n",
        CudaBlocks(), CudaThreads(), CudaThreadIterations(), get_mem_size(false),
        encrypted_data.size(), results_per_batch(), keys_per_batch(), Config().toString().c_str(),
        kernel_inputs.GetLearningMatrix().to_string().c_str());
}


void BruteforceRound::alloc()
{
    //
    assert(kernel_inputs.encdata == nullptr		&& "Encrypted data already allocated on GPU");
    assert(kernel_inputs.decryptors == nullptr	&& "Decryptors data already allocated on GPU");
    assert(kernel_inputs.results == nullptr		&& "Results data already allocated on GPU");

    // ALLOCATE ON GPU
    if (kernel_inputs.encdata == nullptr)
    {
        kernel_inputs.encdata = CudaArray<EncParcel>::allocate(encrypted_data);
    }

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
    if (kernel_inputs.encdata != nullptr)
    {
        kernel_inputs.encdata->free();
        kernel_inputs.encdata = nullptr;
    }

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

    encrypted_data.clear();
}
