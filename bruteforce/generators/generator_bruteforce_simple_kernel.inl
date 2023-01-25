#include "device/cuda_context.h"

#include "kernels/kernel_result.h"
#include "algorithm/keeloq/keeloq_kernel_input.h"
#include "bruteforce/bruteforce_type.h"

__global__ void DEFINE_GENERATOR_KERNEL(GeneratorBruteforceSimple, KeeloqKernelInput::TCudaPtr input, KernelResult::TCudaPtr resuls)
{
	CudaContext ctx = CudaContext::Get();

	assert(input->generator.type == BruteforceType::Simple);

	Decryptor& start = input->generator.start;

	CudaArray<Decryptor>& decryptors = *input->decryptors;

	CUDA_FOR_THREAD_ID(ctx, decryptor_index, decryptors.num)
	{
		Decryptor& decryptor = decryptors[decryptor_index];

		decryptor.man = start.man + decryptor_index;
	}
}

DEFINE_GENERATOR_GETTER(GeneratorBruteforceSimple);