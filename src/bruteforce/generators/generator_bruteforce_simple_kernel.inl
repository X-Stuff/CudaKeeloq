#include "device/cuda_context.h"

#include "kernels/kernel_result.h"
#include "algorithm/keeloq/keeloq_kernel_input.h"
#include "bruteforce/bruteforce_type.h"

__global__ void DEFINE_GENERATOR_KERNEL(GeneratorBruteforceSimple, KeeloqKernelInput::TCudaPtr input, KernelResult::TCudaPtr resuls)
{
	CudaContext ctx = CudaContext::Get();

	assert(input->config.type == BruteforceType::Simple);

	Decryptor& start = input->config.start;

	CudaArray<Decryptor>& decryptors = *input->decryptors;

	CUDA_FOR_THREAD_ID(ctx, decryptor_index, decryptors.num)
	{
		decryptors[decryptor_index] = Decryptor(start.man() + decryptor_index, start.seed());
	}
}

DEFINE_GENERATOR_GETTER(GeneratorBruteforceSimple);