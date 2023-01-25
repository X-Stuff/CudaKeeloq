#include "common.h"
#include "test_filters.h"

#include <cuda_runtime_api.h>

__global__ void Kernel_RunFiltersTests(BruteforceFiltersTestInputs* tests, uint8_t num)
{
	for (int i = 0; i < num; ++i)
	{
		bool value = BruteforceFilters::check_filters(tests[i].value, tests[i].flags);
		assert(value == tests[i].result);

		tests[i].value = value == tests[i].result;
	}
}

void Kernel_LaunchFiltersTests(BruteforceFiltersTestInputs * tests, uint8_t num)
{
	void* args[] = { &tests, &num };
	auto error = cudaLaunchKernel(&Kernel_RunFiltersTests, dim3(), dim3(), args, 0, nullptr);
	CUDA_CHECK(error);
}