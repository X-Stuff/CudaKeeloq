#pragma once

#include "keeloq_types.cuh"

__global__ void CUDA_keeloq_generate_brute(KernelInput::TCudaPtr input, KernelResult::TCudaPtr resuls);



int CUDA_generator_wrapper(KernelInput& mainInputs, uint16_t ThreadBlocks, uint16_t ThreadsInBlock);

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace generators
{
namespace tests
{
void run();
}
}