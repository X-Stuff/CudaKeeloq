#include "doctest/doctest.h"

#include <cuda_runtime_api.h>

#include "device/cuda_double_array.h"
#include "device/cuda_vector.h"

#include "bruteforce/bruteforce_filters.h"
#include "bruteforce/bruteforce_config.h"
#include "bruteforce/generators/generator_bruteforce.h"

#include "tests/support/bruteforce_filter_case.h"
#include "tests/support/keeloq_inputs.h"


namespace
{
const BruteforceFiltersTestInputs kFilterCases[] = {
    { 0x1111334404bbccee, BruteforceFilters::Flags::Max6ZerosInARow, true  },
    { 0x11113344aabbccee, BruteforceFilters::Flags::Max6ZerosInARow, false },
    { 0x11113344aabbccee, BruteforceFilters::Flags::Max6OnesInARow,  false },
    { 0x11113344aFFbccee, BruteforceFilters::Flags::Max6OnesInARow,  true  },

    { 0x3132333435363738, BruteforceFilters::Flags::AsciiNumbers,  true  },
    { 0x3132333435363738, BruteforceFilters::Flags::AsciiAlphaNum, true  },
    { 0x3132333435363738, BruteforceFilters::Flags::AsciiAny,      true  },
    { 0x2931323334353637, BruteforceFilters::Flags::AsciiNumbers,  false },

    { 0x4142434465666768, BruteforceFilters::Flags::AsciiAlpha,    true  },
    { 0x4142434465666768, BruteforceFilters::Flags::AsciiAlphaNum, true  },
    { 0x4142434465666768, BruteforceFilters::Flags::AsciiAny,      true  },
    { 0x3142434465666768, BruteforceFilters::Flags::AsciiAlpha,    false },
    { 0x2142434465666768, BruteforceFilters::Flags::AsciiAlphaNum, false },
    { 0x1142434465666768, BruteforceFilters::Flags::AsciiAny,      false },

    { 0x214023245e28297e, BruteforceFilters::Flags::AsciiSpecial,  true  },
    { 0x214023245e28297e, BruteforceFilters::Flags::AsciiAny,      true  },
    { 0x114023245e28297e, BruteforceFilters::Flags::AsciiSpecial,  false },

    { 0x0022222222556677, BruteforceFilters::Flags::BytesRepeat4,  true  },
    { 0x0022222222226677, BruteforceFilters::Flags::BytesRepeat4,  true  },
    { 0x00Abcdef11111111, BruteforceFilters::Flags::BytesRepeat4,  true  },
    { 0x0011222222556677, BruteforceFilters::Flags::BytesRepeat4,  false },
    { 0x0011223344556677, BruteforceFilters::Flags::BytesRepeat4,  false },

    { 0x112233445566aa00, BruteforceFilters::Flags::BytesIncremental, true  },
    { 0xFFEEDDCCBBAA1234, BruteforceFilters::Flags::BytesIncremental, true  },
    { 0x1122334455778899, BruteforceFilters::Flags::BytesIncremental, false },
};

constexpr uint8_t kFilterCaseCount = sizeof(kFilterCases) / sizeof(kFilterCases[0]);
}


TEST_CASE("filters: CPU check_filters matches expected results")
{
    for (uint8_t i = 0; i < kFilterCaseCount; ++i)
    {
        CAPTURE(i);
        CAPTURE(kFilterCases[i].value);
        CAPTURE(kFilterCases[i].flags);

        const bool value = BruteforceFilters::check_filters(kFilterCases[i].value, kFilterCases[i].flags);
        CHECK(value == kFilterCases[i].result);
    }
}

TEST_CASE("filters: GPU check_filters agrees with CPU for every case")
{
    BruteforceFiltersTestInputs cases[kFilterCaseCount];
    for (uint8_t i = 0; i < kFilterCaseCount; ++i)
    {
        cases[i] = kFilterCases[i];
    }

    DoubleArray<BruteforceFiltersTestInputs> test_inputs(cases, kFilterCaseCount);
    tests::cuda_check_bruteforce_filters(test_inputs.CUDA_mem, kFilterCaseCount);
    test_inputs.readGpu();

    for (uint8_t i = 0; i < kFilterCaseCount; ++i)
    {
        CAPTURE(i);
        CHECK(test_inputs.HOST_mem[i].value == 1);
    }
}

TEST_CASE("filters: filtered generator produces the target filtered key")
{
    const CudaConfig config = CudaConfig::Tests();

    constexpr auto NumToGenerate = 0xFFFFF;
    constexpr auto FilteredKey = 0xAADEADBEEFA63ED2;
    constexpr auto NumInputs = 3;

    const auto inputsTransform = InputsTransform::None;
    auto first_decryptor = Decryptor::Make(0xAADEADBEEFA00000, 0, true);

    auto testConfig = BruteforceConfig::GetBruteforce(first_decryptor, inputsTransform, NumToGenerate,
        BruteforceFilters{
            BruteforceFilters::Flags::All,
            BruteforceFilters::Flags::BytesIncremental | BruteforceFilters::Flags::BytesRepeat4,
        });

    auto inputs = tests::keeloq::genInputs(FilteredKey, NumInputs, inputsTransform);

    KeeloqKernelMultiLearningInput generatorInputs;
    generatorInputs.Initialize(testConfig, inputs);
    generatorInputs.AllocateGPU(testConfig.bruteSize(), NumInputs);

    REQUIRE(GeneratorBruteforce::PrepareDecryptors(generatorInputs, config) == cudaSuccess);

    const auto& dcpu = generatorInputs.decryptors->read();

    bool found = false;
    for (size_t i = 0; !found && i < dcpu.size(); ++i)
    {
        found = dcpu[i].man() == FilteredKey;
    }
    CHECK(found);
}
