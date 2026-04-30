#include "device/cuda_context.h"

#include "kernels/kernel_result.h"
#include "algorithm/keeloq/keeloq_kernel_input.h"
#include "bruteforce/bruteforce_type.h"

__global__ void DEFINE_GENERATOR_KERNEL(GeneratorBruteforceSimple, KeeloqKernelInput::TCudaPtr input, KernelResult::TCudaPtr resuls)
{
    CudaContext ctx = CudaContext::Get();

    assert(input->GetConfig().type == BruteforceType::Simple);

    const Decryptor& start = input->GetConfig().start;

    CudaArray<Decryptor>& decryptors = *input->decryptors;
    assert(decryptors.num % ctx.thread_max == 0 && "Number of decryptors must be equal or divisible by number of threads");

    CUDA_FOR_THREAD_ID(ctx, decryptor_index, decryptors.num)
    {
        decryptors[decryptor_index] = Decryptor::Make(start.man() + decryptor_index, start.seed(), start.has_seed());
    }
}

DEFINE_GENERATOR_GETTER(GeneratorBruteforceSimple);