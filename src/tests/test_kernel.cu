#include "common.h"

#include "test_all.h"
#include "device/cuda_common.h"

#include <cuda_runtime_api.h>

namespace
{

__host__ __device__ void check_rev_utils(uint64_t val1, uint64_t* rev_value1, uint64_t val2, uint64_t* rev_value2, bool rev_bits)
{
    if (rev_bits)
    {
        *rev_value1 = misc::rev_bits(val1);
        *rev_value2 = misc::rev_bits(val2);
    }
    else
    {
        *rev_value1 = misc::rev_bytes(val1);
        *rev_value2 = misc::rev_bytes(val2);
    }
}

__global__ void Kernel_test_rev_utils(uint64_t val1, uint64_t val2, uint64_t* resutls)
{
    check_rev_utils(val1, &resutls[0], val2, &resutls[1], true);
    check_rev_utils(val1, &resutls[2], val2, &resutls[3], false);
}

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

__host__ void tests::cuda_check_bruteforce_filters(BruteforceFiltersTestInputs * tests, uint8_t num)
{
    void* args[] = { &tests, &num };
    auto error = cudaLaunchKernel((void*) & Kernel_RunFiltersTests, dim3(), dim3(), args, 0, nullptr);
    CUDA_CHECK(error);
}

__host__ bool tests::check_utils()
{
    // Test Case 1: Patterned Hex
    uint64_t val1 = 0x123456789ABCDEF0ULL;
    uint64_t val2 = 0x0000000000000001ULL;

    uint64_t bit_rev1, bit_rev2;
    uint64_t byte_rev1, byte_rev2;

    check_rev_utils(val1, &bit_rev1, val2, &bit_rev2, true);
    check_rev_utils(val1, &byte_rev1, val2, &byte_rev2, false);

    // CPU
    assert(bit_rev1 == 0x0F7B3D591E6A2C48ULL);
    assert(bit_rev2 == 0x8000000000000000ULL);
    assert(byte_rev1 == 0xF0DEBC9A78563412ULL);
    assert(byte_rev2 == 0x0100000000000000ULL);

    // GPU - alloc memory for results
    uint64_t* cuda_results = nullptr;
    uint64_t cpu_results[4] = { 0 };
    auto error = cudaMalloc((void**)&cuda_results, sizeof(uint64_t) * 4);
    CUDA_CHECK(error);

    Kernel_test_rev_utils<<<1, 1 >>>(val1, val2, cuda_results);

    error = cudaMemcpy(cpu_results, cuda_results, sizeof(cpu_results), cudaMemcpyDeviceToHost);
    CUDA_CHECK(error);

    error = cudaFree(cuda_results);
    CUDA_CHECK(error);

    // GPU
    assert(cpu_results[0] == 0x0F7B3D591E6A2C48ULL);
    assert(cpu_results[1] == 0x8000000000000000ULL);
    assert(cpu_results[2] == 0xF0DEBC9A78563412ULL);
    assert(cpu_results[3] == 0x0100000000000000ULL);

    return bit_rev1 == cpu_results[0] && bit_rev2 == cpu_results[1] && byte_rev1 == cpu_results[2] && byte_rev2 == cpu_results[3];
}

__host__ bool tests::cuda_check_working()
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