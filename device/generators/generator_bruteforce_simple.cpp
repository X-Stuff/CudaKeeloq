#include "generator_bruteforce_simple.h"

#include "device/cuda_context.h"

USE_NS_LOCATION

__global__ void Kernel_GeneratorBruteforceSimple(KernelInput::TCudaPtr input, KernelResult::TCudaPtr resuls)
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

GeneratorBruteforceSimple::KernelFunc GeneratorBruteforceSimple::GetKernelFunctionPtr()
{
	return &Kernel_GeneratorBruteforceSimple;
}
