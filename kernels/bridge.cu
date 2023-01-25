#include "bridge.h"

#include "keeloq_main.cuh"

namespace
{
	__global__ void Kernel_Test(uint32_t* input)
	{
		*input = 42;
	}
}

bool Bridge::Kernel_CheckBridgeIsWorking()
{
	uint32_t result = 0;
	uint32_t* pInput;

	auto error = cudaMalloc((void**)&pInput, sizeof(uint32_t));
	CUDA_CHECK(error);

	void *args[] = { &pInput };
	auto* func = &Kernel_Test;
	error = cudaLaunchKernel(func, dim3(), dim3(), args, 0, 0);
	CUDA_CHECK(error);

	error = cudaMemcpy(&result, pInput, sizeof(uint32_t), cudaMemcpyDeviceToHost);
	CUDA_CHECK(error);

	assert(result == 42);
	Kernel_Test<<<1, 1>>>(pInput);

	error = cudaFree(pInput);
	CUDA_CHECK(error);

	return result == 42;
}

bool Bridge::Kernel_CheckKeeloqIsWorking()
{
	return CUDA_check_keeloq_works();
}

void Bridge::Kernel_KeeloqBruteMain(KernelInput& inputs, KernelResult& result, uint32_t blocks, uint32_t threads)
{
	result = std::move(CUDA_keeloq_main_wrapper(inputs, blocks, threads));
}
