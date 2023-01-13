#pragma once

#include "keeloq_main.cuh"

__global__ void CUDA_keeloq_generate_brute(KernelInput::TCudaPtr input, KernelResult::TCudaPtr resuls);

__global__ void CUDA_keeloq_generate_smart(KernelInput::TCudaPtr input, KernelResult::TCudaPtr resuls);


template<uint16_t ThreadBlocks, uint16_t ThreadsInBlock>
int CUDA_generator_wrapper(KernelInput& mainInputs)
{
    KernelResult generator_results;

    switch (mainInputs.generator.type)
    {

    case GeneratorType::Brute:
        CUDA_keeloq_generate_brute<<<ThreadBlocks, ThreadsInBlock>>>(mainInputs.ptr(), generator_results.ptr());
        break;
    case GeneratorType::Smart:
        CUDA_keeloq_generate_smart<<<ThreadBlocks, ThreadsInBlock>>>(mainInputs.ptr(), generator_results.ptr());
        break;

    case GeneratorType::Dictionary:
    default:
        return 0;
    }

    mainInputs.read(); // it will not cause underneath arrays copy
    generator_results.read();

    return generator_results.error;
}