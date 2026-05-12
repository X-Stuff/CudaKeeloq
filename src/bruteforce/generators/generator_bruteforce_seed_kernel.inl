#include "device/cuda_context.h"

#include "kernels/kernel_result.h"
#include "kernels/kernel_input_multi_learning.h"

#include "bruteforce/bruteforce_type.h"

__global__ void DEFINE_GENERATOR_KERNEL(GeneratorBruteforceSeed, IKeeloqKernelInputBase::Ptr input, KernelResult::TCudaPtr resuls)
{
    CudaContext ctx = CudaContext::Get();

    assert(input->GetConfig().type == BruteforceType::Seed || input->GetConfig().type == BruteforceType::XorFix);

    const Decryptor& start = input->GetConfig().start;
    static constexpr bool seed_valid = true;

    CudaArray<Decryptor>& decryptors = *input->decryptors;
    assert(decryptors.num % ctx.thread_max == 0 && "Number of decryptors must be equal or divisible by number of threads");

    CUDA_FOR_THREAD_ID(ctx, decryptor_index, decryptors.num)
    {
        decryptors[decryptor_index] = Decryptor::Make(start.man(), start.seed() + decryptor_index, seed_valid);
    }
}

DEFINE_GENERATOR_GETTER(GeneratorBruteforceSeed);