#pragma once

#include "keeloq_main.cuh"

__host__ __device__ bool check_filters(uint64_t key, SmartFilterFlags filter);

__global__ void CUDA_keeloq_generate_brute(KernelInput::TCudaPtr input, KernelResult::TCudaPtr resuls);

__global__ void CUDA_keeloq_generate_smart(KernelInput::TCudaPtr input, KernelResult::TCudaPtr resuls);


__global__ void CUDA_generators_test(KernelResult::TCudaPtr resuls);


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

inline int CUDA_test_generator_filters()
{
    KernelResult test_results;

    struct test_input
    {
        uint64_t value;
        SmartFilterFlags flags;
        bool result;
    };

    test_input test_cases[] = {
        { 0x1111334404bbccee, SmartFilterFlags::Max6ZerosInARow, true },
        { 0x11113344aabbccee, SmartFilterFlags::Max6ZerosInARow, false },
        { 0x11113344aabbccee, SmartFilterFlags::Max6OnesInARow, false },
        { 0x11113344aFFbccee, SmartFilterFlags::Max6OnesInARow, true },

        { 0x3132333435363738, SmartFilterFlags::AsciiNumbers, true },
        { 0x3132333435363738, SmartFilterFlags::AsciiAlphaNum, true },
        { 0x3132333435363738, SmartFilterFlags::AsciiAnySymbol, true },
        { 0x2931323334353637, SmartFilterFlags::AsciiNumbers, false },

        { 0x4142434465666768, SmartFilterFlags::AsciiAlpha,     true },
        { 0x4142434465666768, SmartFilterFlags::AsciiAlphaNum,  true },
        { 0x4142434465666768, SmartFilterFlags::AsciiAnySymbol, true },
        { 0x3142434465666768, SmartFilterFlags::AsciiAlpha,     false },
        { 0x2142434465666768, SmartFilterFlags::AsciiAlphaNum,  false },
        { 0x1142434465666768, SmartFilterFlags::AsciiAnySymbol, false },

        { 0x214023245e28297e, SmartFilterFlags::AsciiSpecial, true },
        { 0x214023245e28297e, SmartFilterFlags::AsciiAnySymbol, true },
        { 0x114023245e28297e, SmartFilterFlags::AsciiSpecial, false },
    };

    for (int i = 0; i < sizeof(test_cases) / sizeof(test_input); ++i)
    {
        bool value = check_filters(test_cases[i].value, test_cases[i].flags);
        assert(value == test_cases[i].result);
    }


    //CUDA_generators_test<<<1,1>>>(test_results.ptr());

    test_results.read();
    return test_results.error;
}