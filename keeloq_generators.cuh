#pragma once

#include "keeloq_types.cuh"

__host__ __device__ bool check_filters(uint64_t key, SmartFilterFlags filter);

__global__ void CUDA_keeloq_generate_brute(KernelInput::TCudaPtr input, KernelResult::TCudaPtr resuls);

__global__ void CUDA_keeloq_generate_filtered(KernelInput::TCudaPtr input, KernelResult::TCudaPtr resuls);

__global__ void CUDA_keeloq_generate_alphabet(KernelInput::TCudaPtr input, KernelResult::TCudaPtr resuls);

__global__ void CUDA_generators_filters_test(struct FiltersTestinput* tests, uint8_t num);


int CUDA_generator_wrapper(KernelInput& mainInputs, uint16_t ThreadBlocks, uint16_t ThreadsInBlock);

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct FiltersTestinput
{
    uint64_t value;
    SmartFilterFlags flags;
    bool result;
};

namespace generators
{
namespace tests
{
void run();
}
}