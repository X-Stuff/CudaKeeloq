#include "generator_bruteforce.h"

#include "algorithm/keeloq/keeloq_kernel_input.h"
#include "kernels/kernel_result.h"


int GeneratorBruteforce::PrepareDecryptors(KeeloqKernelInput& inputs, uint16_t blocks, uint16_t threads)
{
    KernelResult generator_results;

    switch (inputs.config.type)
    {
    case BruteforceType::Simple:
    {
        GeneratorBruteforceSimple::LaunchKernel(blocks, threads, inputs.ptr(), generator_results.ptr());
        break;
    }
    case BruteforceType::Filtered:
    {
        inputs.config.filters.sync_key = inputs.config.start.man();
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
        return 0;
    }

    inputs.read();          // it will not cause underneath arrays copy
    generator_results.read();

    // last generated decryptor - is first on next batch
    //  Warning: In case of non-aligned calculations "real" last decryptor may be somewhere in the middle of array
    inputs.config.last = inputs.decryptors->host_last();

    return generator_results.error;
}
