#include "common.h"

#include "test_all.h"
#include "device/cuda_common.h"

#include <cuda_runtime_api.h>

namespace
{
    __global__ void Kernel_Test(uint64_t* input)
    {
        *input = 0x1234567890ABCDEF;
        *input = misc::rev_bytes(*input);
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
    auto error = cudaLaunchKernel((void*) & Kernel_RunFiltersTests, dim3(), dim3(), args, 0, nullptr);
    CUDA_CHECK(error);
}

__host__ bool Tests::CheckCudaIsWorking()
{
    uint64_t result = 0;
    uint64_t* pInput;

    auto error = cudaMalloc((void**)&pInput, sizeof(uint64_t));
    CUDA_CHECK(error);

    void *args[] = { &pInput };
    auto* func = (void*) &Kernel_Test;
    error = cudaLaunchKernel(func, dim3(), dim3(), args, 0, 0);
    CUDA_CHECK(error);

    error = cudaMemcpy(&result, pInput, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    CUDA_CHECK(error);

    assert(result == 0xEFCDAB9078563412);
    Kernel_Test<<<1, 1>>>(pInput);

    error = cudaFree(pInput);
    CUDA_CHECK(error);

    return result == 0xEFCDAB9078563412;
}