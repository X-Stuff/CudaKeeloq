#include "common.h"

#include "test_all.h"

#include <cuda_runtime_api.h>

namespace
{
	__global__ void Kernel_Test(uint32_t* input)
	{
		*input = 42;
	}

	__global__ void Kernel_RunFiltersTests(BruteforceFiltersTestInputs* tests, uint8_t num)
	{
		for (int i = 0; i < num; ++i)
		{
			bool value = BruteforceFilters::check_filters(tests[i].value, tests[i].flags);
			assert(value == tests[i].result);

			tests[i].value = value == tests[i].result;
		}
	}
}

__host__ void Tests::LaunchFiltersTests(BruteforceFiltersTestInputs * tests, uint8_t num)
{
	void* args[] = { &tests, &num };
	auto error = cudaLaunchKernel(&Kernel_RunFiltersTests, dim3(), dim3(), args, 0, nullptr);
	CUDA_CHECK(error);
}

__host__ bool Tests::CheckCudaIsWorking()
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