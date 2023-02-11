#include "test_pattern.h"

#include <algorithm>

#include "bruteforce/bruteforce_config.h"
#include "bruteforce/bruteforce_pattern.h"

#include "device/cuda_vector.h"

#include "algorithm/keeloq/keeloq_kernel.h"
#include "algorithm/keeloq/keeloq_kernel_input.h"

#include "bruteforce/generators/generator_bruteforce.h"


namespace
{
BruteforceConfig GetSingleKeyConfig(bool rev = true)
{
    std::vector<std::vector<uint8_t>> pattern =
    {
        { 0xCE }, { 0xB6 }, { 0xAE }, { 0x48 }, { 0xB5 }, { 0xC6 }, { 0x3E }, { 0xD2 },
    };

    if (rev)
    {
        std::reverse(pattern.begin(), pattern.end());
    }

    BruteforcePattern br_pattern(std::move(pattern), "Test");
    return BruteforceConfig::GetPattern(0x0, br_pattern, 0xFFFFFFFF);
}
}


bool Tests::PatternGeneration()
{
    constexpr auto NumBlocks = 64;
    constexpr auto NumThreads = 64;

    CudaVector<EncData> encrypted  =
    {
        0xC65D52A0A81FD504,0xCCA9B335A81FD504,0xE0DA7372A81FD504
    };

    CudaVector<Decryptor> decryptors(NumBlocks * NumThreads);
    CudaVector<SingleResult> results(decryptors.size() * encrypted.size());

    BruteforceConfig config = GetSingleKeyConfig();
    auto init = config.pattern.init(0);
    assert(init.number() == 0xCEB6AE48B5C63ED2);

    KeeloqKernelInput generatorInputs(encrypted.gpu(), decryptors.gpu(), results.gpu(), config);

    GeneratorBruteforce::PrepareDecryptors(generatorInputs, NumBlocks, NumThreads);
    auto result = keeloq::kernels::BruteMain(generatorInputs, NumBlocks, NumThreads);

    decryptors.read();
    results.read();
    assert(decryptors.cpu()[0].man == 0xCEB6AE48B5C63ED2);

    return decryptors.cpu()[0].man == 0xCEB6AE48B5C63ED2;
}

