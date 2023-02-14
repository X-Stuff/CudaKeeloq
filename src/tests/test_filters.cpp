#include "tests/test_filters.h"

#include <cuda_runtime_api.h>

#include "device/cuda_vector.h"
#include "device/cuda_double_array.h"

#include "bruteforce/bruteforce_filters.h"
#include "bruteforce/generators/generator_bruteforce.h"


bool tests::filters_generation()
    {
        BruteforceFiltersTestInputs test_cases[] = {
           { 0x1111334404bbccee, BruteforceFilters::Flags::Max6ZerosInARow, true },
           { 0x11113344aabbccee, BruteforceFilters::Flags::Max6ZerosInARow, false },
           { 0x11113344aabbccee, BruteforceFilters::Flags::Max6OnesInARow, false },
           { 0x11113344aFFbccee, BruteforceFilters::Flags::Max6OnesInARow, true },

           { 0x3132333435363738, BruteforceFilters::Flags::AsciiNumbers, true },
           { 0x3132333435363738, BruteforceFilters::Flags::AsciiAlphaNum, true },
           { 0x3132333435363738, BruteforceFilters::Flags::AsciiAny, true },
           { 0x2931323334353637, BruteforceFilters::Flags::AsciiNumbers, false },

           { 0x4142434465666768, BruteforceFilters::Flags::AsciiAlpha,     true },
           { 0x4142434465666768, BruteforceFilters::Flags::AsciiAlphaNum,  true },
           { 0x4142434465666768, BruteforceFilters::Flags::AsciiAny, true },
           { 0x3142434465666768, BruteforceFilters::Flags::AsciiAlpha,     false },
           { 0x2142434465666768, BruteforceFilters::Flags::AsciiAlphaNum,  false },
           { 0x1142434465666768, BruteforceFilters::Flags::AsciiAny, false },

           { 0x214023245e28297e, BruteforceFilters::Flags::AsciiSpecial, true },
           { 0x214023245e28297e, BruteforceFilters::Flags::AsciiAny, true },
           { 0x114023245e28297e, BruteforceFilters::Flags::AsciiSpecial, false },

           { 0x0022222222556677, BruteforceFilters::Flags::BytesRepeat4, true },
           { 0x0022222222226677, BruteforceFilters::Flags::BytesRepeat4, true },
           { 0x00Abcdef11111111, BruteforceFilters::Flags::BytesRepeat4, true },
           { 0x0011222222556677, BruteforceFilters::Flags::BytesRepeat4, false },
           { 0x0011223344556677, BruteforceFilters::Flags::BytesRepeat4, false },

           { 0x112233445566aa00, BruteforceFilters::Flags::BytesIncremental, true },
           { 0xFFEEDDCCBBAA1234, BruteforceFilters::Flags::BytesIncremental, true },
           { 0x1122334455778899, BruteforceFilters::Flags::BytesIncremental, false },
        };

        static uint8_t NumTests = sizeof(test_cases) / sizeof(BruteforceFiltersTestInputs);
        bool result_success = true;

        // CPU tests
        for (int i = 0; i < NumTests; ++i)
        {
            bool value = BruteforceFilters::check_filters(test_cases[i].value, test_cases[i].flags);
            result_success &= value == test_cases[i].result;

            assert(result_success);
        }

        // GPU tests
        DoubleArray<BruteforceFiltersTestInputs> test_inputs(test_cases, NumTests);
        cuda_check_bruteforce_filters(test_inputs.CUDA_mem, NumTests);
        test_inputs.read_GPU(); // for asserts

        for (uint8_t i = 0; i < NumTests; ++i)
        {
            result_success &= test_inputs.HOST_mem[i].value == 1;
            assert(result_success);
        }

        // Filtered generator test itself
        constexpr auto NumBlocks = 32;// 1; //
        constexpr auto NumThreads = 512;// 2;// 512;

        constexpr auto NumToGenerate = 0xFFFFF;

        auto testConfig = BruteforceConfig::GetBruteforce(0xAADEADBEEFA00000, NumToGenerate,
            BruteforceFilters{
                BruteforceFilters::Flags::All,     // SmartFilterFlags::AsciiAny;       //
                BruteforceFilters::Flags::BytesIncremental | BruteforceFilters::Flags::BytesRepeat4,    // SmartFilterFlags::BytesRepeat4;   //
            });

        CudaVector<Decryptor> decryptors(NumToGenerate);

        KeeloqKernelInput generatorInputs(nullptr, decryptors.gpu(), nullptr, testConfig);
        KernelResult result;

        auto error = GeneratorBruteforce::PrepareDecryptors(generatorInputs, NumBlocks, NumThreads);
        result_success &= error == 0;

        decryptors.read();

        bool found = false;
        for (size_t i = 0; !found && i < decryptors.size(); ++i)
        {
            // looking for exact code - check nothing missed
            found |= decryptors.cpu()[i].man == 0xAADEADBEEFA63ED2;
        }

        assert(found);
        result_success &= found;

        result.read();
        return result_success;
    }
