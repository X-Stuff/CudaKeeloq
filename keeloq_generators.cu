#include "keeloq_generators.cuh"

__global__ void CUDA_keeloq_generate_brute(KernelInput::TCudaPtr input, KernelResult::TCudaPtr resuls)
{
	CUDACtx ctx = GET_CUDA_CONTEXT();

	assert(input->generator.type == GeneratorType::Brute);

	Decryptor& start = input->generator.start;
	Decryptor& next = input->generator.next;

	CUDA_Array<Decryptor>& decryptors = *input->decryptors;

	CUDA_FOR_THREAD_ID(ctx, decryptor_index, decryptors.num)
	{
		Decryptor& decryptor = decryptors[decryptor_index];

		decryptor.man = start.man + decryptor_index;
	}

	// every thread will do this. need to measure perf
	next.man = decryptors[decryptors.num - 1].man;
}

__global__ void CUDA_keeloq_generate_smart(KernelInput::TCudaPtr input, KernelResult::TCudaPtr resuls)
{
	CUDACtx ctx = GET_CUDA_CONTEXT();

	assert(input->generator.type == GeneratorType::Smart);

	CUDA_FOR_THREAD_ID(ctx, decryptor_index, input->decryptors->num)
	{

	}
}
