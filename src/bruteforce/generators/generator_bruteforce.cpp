#include "generator_bruteforce.h"

#include "algorithm/keeloq/keeloq_kernel_input.h"
#include "kernels/kernel_result.h"


int GeneratorBruteforce::PrepareDecryptors(KeeloqKernelInput& inputs, uint16_t blocks, uint16_t threads)
{
    const BruteforceConfig& config = inputs.GetConfig();
    KernelResult generator_results;

    inputs.BeforeGenerateDecryptors();

    switch (config.type)
    {
    case BruteforceType::Simple:
    {
        GeneratorBruteforceSimple::LaunchKernel(blocks, threads, inputs.ptr(), generator_results.ptr());
        break;
    }
    case BruteforceType::Seed:
    {
        GeneratorBruteforceSeed::LaunchKernel(blocks, threads, inputs.ptr(), generator_results.ptr());
        break;
    }
    case BruteforceType::Filtered:
    {
        GeneratorBruteforceFiltered::LaunchKernel(blocks, threads, inputs.ptr(), generator_results.ptr());
        break;
    }
    case BruteforceType::Pattern:
    case BruteforceType::Alphabet:
    {
        GeneratorBruteforcePattern::LaunchKernel(blocks, threads, inputs.ptr(), generator_results.ptr());
        break;
    }
    case BruteforceType::Dictionary:
    {
        return 0;
    }
    default:

        printf("Error: Invalid bruteforce type: %d %s! Don't know how to generate decryptors!\n",
            (int)config.type, BruteforceType::Name(config.type));
        return 0;
    }

    inputs.read();          // it will not cause underneath arrays copy
    generator_results.read();

    inputs.AfterGeneratedDecryptors();

    return generator_results.error;
}
