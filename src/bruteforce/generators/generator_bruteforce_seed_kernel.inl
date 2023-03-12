#include "device/cuda_context.h"

#include "kernels/kernel_result.h"
#include "algorithm/keeloq/keeloq_kernel_input.h"
#include "bruteforce/bruteforce_type.h"

__global__ void DEFINE_GENERATOR_KERNEL(GeneratorBruteforceSeed, KeeloqKernelInput::TCudaPtr input, KernelResult::TCudaPtr resuls)
{
	CudaContext ctx = CudaContext::Get();

	assert(input->GetConfig().type == BruteforceType::Seed);

	const Decryptor& start = input->GetConfig().start;

	CudaArray<Decryptor>& decryptors = *input->decryptors;

	CUDA_FOR_THREAD_ID(ctx, decryptor_index, decryptors.num)
	{
		decryptors[decryptor_index] = Decryptor(start.man(), start.seed() + decryptor_index);
	}
}

DEFINE_GENERATOR_GETTER(GeneratorBruteforceSeed);