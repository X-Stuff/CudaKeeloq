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

inline int CUDA_test_generator_filters()
{
    FiltersTestinput test_cases[] = {
        { 0x1111334404bbccee, SmartFilterFlags::Max6ZerosInARow, true },
        { 0x11113344aabbccee, SmartFilterFlags::Max6ZerosInARow, false },
        { 0x11113344aabbccee, SmartFilterFlags::Max6OnesInARow, false },
        { 0x11113344aFFbccee, SmartFilterFlags::Max6OnesInARow, true },

        { 0x3132333435363738, SmartFilterFlags::AsciiNumbers, true },
        { 0x3132333435363738, SmartFilterFlags::AsciiAlphaNum, true },
        { 0x3132333435363738, SmartFilterFlags::AsciiAny, true },
        { 0x2931323334353637, SmartFilterFlags::AsciiNumbers, false },

        { 0x4142434465666768, SmartFilterFlags::AsciiAlpha,     true },
        { 0x4142434465666768, SmartFilterFlags::AsciiAlphaNum,  true },
        { 0x4142434465666768, SmartFilterFlags::AsciiAny, true },
        { 0x3142434465666768, SmartFilterFlags::AsciiAlpha,     false },
        { 0x2142434465666768, SmartFilterFlags::AsciiAlphaNum,  false },
        { 0x1142434465666768, SmartFilterFlags::AsciiAny, false },

        { 0x214023245e28297e, SmartFilterFlags::AsciiSpecial, true },
        { 0x214023245e28297e, SmartFilterFlags::AsciiAny, true },
        { 0x114023245e28297e, SmartFilterFlags::AsciiSpecial, false },

        { 0x0022222222556677, SmartFilterFlags::BytesRepeat4, true },
        { 0x0022222222226677, SmartFilterFlags::BytesRepeat4, true },
        { 0x00Abcdef11111111, SmartFilterFlags::BytesRepeat4, true },
        { 0x0011222222556677, SmartFilterFlags::BytesRepeat4, false },
        { 0x0011223344556677, SmartFilterFlags::BytesRepeat4, false },

        { 0x112233445566aa00, SmartFilterFlags::BytesIncremental, true },
        { 0xFFEEDDCCBBAA1234, SmartFilterFlags::BytesIncremental, true },
        { 0x1122334455778899, SmartFilterFlags::BytesIncremental, false },
    };

    constexpr uint8_t NumTests = sizeof(test_cases) / sizeof(FiltersTestinput);

    // CPU tests
    for (int i = 0; i < NumTests; ++i)
    {
        bool value = check_filters(test_cases[i].value, test_cases[i].flags);
        assert(value == test_cases[i].result);
    }

    // GPU tests
    DOUBLE_ARRAY<FiltersTestinput> testInputs(test_cases, NumTests);
    CUDA_generators_filters_test<<<1,1>>>(testInputs.CUDA_mem, NumTests);
    testInputs.read_GPU(); // for asserts

    // Filtered generator test itself
    constexpr auto NumBlocks = 16;
    constexpr auto NumThreads = 32;

    auto testConfig = BruteforceConfig::GetBruteforce(0xAbcdef11111100, 0xFFFFFFFF,
        BruteforceConfig::Filters {
            SmartFilterFlags::All,     // SmartFilterFlags::AsciiAny;       //
            SmartFilterFlags::None,    // SmartFilterFlags::BytesRepeat4;   //
        });

    std::vector<Decryptor> decryptors(NumBlocks * NumThreads);
    KernelInput generatorInputs(nullptr, CUDA_Array<Decryptor>::allocate(decryptors), nullptr, testConfig);
    KernelResult result;

    CUDA_keeloq_generate_filtered<<<NumBlocks,NumThreads>>>(generatorInputs.ptr(), result.ptr());

    generatorInputs.read();
    generatorInputs.decryptors->copy(decryptors);

    result.read();

    return 0;
}

inline int CUDA_test_generator_alphabet()
{
    // Filtered generator test itself
    constexpr auto NumBlocks = 16;
    constexpr auto NumThreads = 32;


    auto testConfig = BruteforceConfig::GetAlphabet(0x6262626262626262, "abcd"_b, 0xFFFFFFFF);

    std::vector<Decryptor> decryptors(NumBlocks * NumThreads);
    KernelInput generatorInputs(nullptr, CUDA_Array<Decryptor>::allocate(decryptors), nullptr, testConfig);
    KernelResult result;

    CUDA_keeloq_generate_alphabet<<<NumBlocks,NumThreads>>>(generatorInputs.ptr(), result.ptr());

    generatorInputs.read();
    generatorInputs.decryptors->copy(decryptors);

    result.read();

    return 0;
}