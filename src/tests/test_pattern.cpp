#include "test_pattern.h"

#include <algorithm>

#include "tests/test_keeloq.h"

#include "bruteforce/bruteforce_config.h"
#include "bruteforce/bruteforce_pattern.h"

#include "device/cuda_vector.h"

#include "algorithm/keeloq/keeloq_kernel.h"
#include "algorithm/keeloq/keeloq_kernel_input.h"

#include "bruteforce/generators/generator_bruteforce.h"


namespace
{
BruteforceConfig GetSingleKeyConfig(uint64_t key, bool rev = true)
{
    uint8_t* pKey = (uint8_t*)&key;

    std::vector<std::vector<uint8_t>> pattern =
    {
        { pKey[7] }, { pKey[6] }, { pKey[5] }, { pKey[4] }, { pKey[3] }, { pKey[2] }, { pKey[1] }, { pKey[0] },
    };

    if (rev)
    {
        std::reverse(pattern.begin(), pattern.end());
    }

    BruteforcePattern br_pattern(std::move(pattern), "Test");
    return BruteforceConfig::GetPattern(Decryptor(0,0), br_pattern, 0xFFFFFFFF);
}
}


bool tests::pattern_generation()
{
    constexpr auto NumBlocks = 64;
    constexpr auto NumThreads = 64;

    const uint64_t debugKey = "hello_world"_u64;

    CudaVector<EncParcel> encrypted  = tests::keeloq::gen_inputs(debugKey);

    CudaVector<Decryptor> decryptors(NumBlocks * NumThreads);
    CudaVector<SingleResult> results(decryptors.size() * encrypted.size());

    BruteforceConfig config = GetSingleKeyConfig(debugKey);
    if (config.pattern.init(0).number() != debugKey)
    {
        assert(false);
        return false;
    }

    KeeloqKernelInput generatorInputs;
    generatorInputs.encdata = encrypted.gpu();
    generatorInputs.decryptors = decryptors.gpu();
    generatorInputs.results = results.gpu();
    generatorInputs.Initialize(config, KeeloqLearningType::full_mask());

    GeneratorBruteforce::PrepareDecryptors(generatorInputs, NumBlocks, NumThreads);
    auto result = ::keeloq::kernels::cuda_brute(generatorInputs, NumBlocks, NumThreads);

    decryptors.read();
    results.read();
    assert(decryptors.cpu()[0].man() == debugKey);

    return decryptors.cpu()[0].man() == debugKey;
}

