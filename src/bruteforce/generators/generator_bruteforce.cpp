#include "generator_bruteforce.h"

#include "algorithm/keeloq/keeloq_kernel_input.h"
#include "kernels/kernel_result.h"
#include "device/cuda_config.h"


cudaError_t GeneratorBruteforce::PrepareDecryptors(KeeloqKernelInput& inputs, const CudaConfig& cuda)
{
    const BruteforceConfig& config = inputs.GetConfig();
    inputs.BeforeGenerateDecryptors();

    switch (config.type)
    {
    case BruteforceType::Simple:
    {
        GeneratorBruteforceSimple::LaunchKernel(cuda, inputs.ptr(), nullptr);
        break;
    }
    case BruteforceType::Seed:
    {
        GeneratorBruteforceSeed::LaunchKernel(cuda, inputs.ptr(), nullptr);
        break;
    }
    case BruteforceType::Filtered:
    {
        GeneratorBruteforceFiltered::LaunchKernel(cuda, inputs.ptr(), nullptr);
        break;
    }
    case BruteforceType::Pattern:
    case BruteforceType::Alphabet:
    {
        GeneratorBruteforcePattern::LaunchKernel(cuda, inputs.ptr(), nullptr);
        break;
    }
    case BruteforceType::Dictionary:
    {
        // TODO: generatorInputs.WriteDecryptors(config.decryptors, 0, config.decryptors.size());
        return cudaSuccess;
    }
    default:
    {
        printf("Error: Invalid bruteforce type: %d (%s)! Don't know how to generate decryptors!\n",
            (int)config.type, BruteforceType::Name(config.type));
        return cudaErrorUnknown;
    }
    }

    inputs.read(); // it will not cause underneath arrays copy
    inputs.AfterGeneratedDecryptors();

    return cudaSuccess;
}
