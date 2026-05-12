#include "device/cuda_context.h"

#include "kernels/kernel_result.h"
#include "kernels/kernel_input_multi_learning.h"

#include "bruteforce/bruteforce_pattern.h"
#include "bruteforce/bruteforce_type.h"


__global__ void DEFINE_GENERATOR_KERNEL(GeneratorBruteforcePattern, IKeeloqKernelInputBase::Ptr input, KernelResult::TCudaPtr resuls)
{
    const BruteforceConfig& config = input->GetConfig();

    assert((config.type == BruteforceType::Alphabet) || (config.type == BruteforceType::Pattern));

    CudaContext ctx = CudaContext::Get();

    const BruteforcePattern& pattern = config.pattern;
    CudaArray<Decryptor>& decryptors = *input->decryptors;
    assert(decryptors.num % ctx.thread_max == 0 && "Number of decryptors must be equal or divisible by number of threads");

    // Imagine alphabet as rotating rings with letters on it
    // we have 8-bytes key so there will be 8 rings
    // bytes in the key show how much ring is rotated
    // and what 'letter' it should have.
    // Or also it can be considered as 8-digit N-based number
    MultibaseNumber start = pattern.init(config.start.man());

    const uint32_t seed = config.start.seed();
    const bool seed_valid = config.start.has_seed();

    CUDA_FOR_THREAD_ID(ctx, decryptor_index, decryptors.num)
    {
        decryptors[decryptor_index] = Decryptor::Make(pattern.next(start, decryptor_index).number(), seed, seed_valid);
    }
}

DEFINE_GENERATOR_GETTER(GeneratorBruteforcePattern);