#include "tests/test_generators.h"
#include "tests/test_keeloq.h"

#include <algorithm>

#include "bruteforce/bruteforce_config.h"
#include "bruteforce/bruteforce_pattern.h"
#include "bruteforce/generators/generator_bruteforce.h"

#include "device/cuda_vector.h"

#include "algorithm/keeloq/keeloq_kernel.h"
#include "algorithm/keeloq/keeloq_kernel_input.h"


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
    return BruteforceConfig::GetPattern(Decryptor::Make(0,0,true), br_pattern, 0xFFFFFFFF);
}
}

bool tests::generators::pattern()
{
    using namespace KeeloqLearning;

    constexpr auto NumBlocks = 64;
    constexpr auto NumThreads = 64;

    const uint64_t debugKey = "hello_world"_u64;

    CudaVector<EncParcel> encrypted  = tests::keeloq::gen_inputs<LearningType::Simple>(debugKey);

    CudaVector<Decryptor> decryptors(NumBlocks * NumThreads);
    CudaVector<SingleResult> results(decryptors.size() * encrypted.size());

    BruteforceConfig config = GetSingleKeyConfig(debugKey);
    assert(config.type == BruteforceType::Pattern && "Invalid BruteforceConfig type, must be Pattern");

    if (config.pattern.init(0).number() != debugKey)
    {
        assert(false && "Invalid pattern initialization, pattern should have debug key as its first number");
        return false;
    }

    KeeloqKernelInput generatorInputs;
    generatorInputs.encdata = encrypted.gpu();
    generatorInputs.decryptors = decryptors.gpu();
    generatorInputs.results = results.gpu();
    generatorInputs.Initialize(config, KeeloqLearning::Matrix::Everything());

    GeneratorBruteforce::PrepareDecryptors(generatorInputs, NumBlocks, NumThreads);
    auto result = ::keeloq::kernels::cuda_brute(generatorInputs, NumBlocks, NumThreads);

    decryptors.read();
    results.read();
    assert(decryptors.cpu()[0].man() == debugKey);

    return decryptors.cpu()[0].man() == debugKey;
}

bool tests::generators::seed()
{
    using namespace KeeloqLearning;

    constexpr auto NumBlocks = 64;
    constexpr auto NumThreads = 64;
    constexpr auto NumTestRounds = 8;

    const uint64_t debugKey = "hello_world"_u64;

    CudaVector<EncParcel> encrypted = tests::keeloq::gen_inputs<LearningType::Secure>(debugKey);
    CudaVector<Decryptor> decryptors(NumBlocks * NumThreads);
    CudaVector<SingleResult> results(decryptors.size() * encrypted.size());

    BruteforceConfig config = BruteforceConfig::GetSeedBruteforce(Decryptor::Make(debugKey, 0, true));
    assert(config.type == BruteforceType::Seed && "Invalid BruteforceConfig type, must be Seed");

    // Initialize brute inputs manually
    KeeloqKernelInput generatorInputs;
    generatorInputs.encdata = encrypted.gpu();
    generatorInputs.decryptors = decryptors.gpu();
    generatorInputs.results = results.gpu();
    generatorInputs.Initialize(config, KeeloqLearning::Matrix::Everything());

    uint64_t gen_seed_global = 0;
    for (int round = 0; round < NumTestRounds; ++round)
    {
        if (round != 0)
        {
            // Make last decryptor from previous batch as first for this batch
            generatorInputs.NextDecryptor();

            // Since last became first, and first is generated as well - we need to reduce global counter by 1
            gen_seed_global -= 1;
        }

        // This will generate decryptors in GPU memory
        auto error = GeneratorBruteforce::PrepareDecryptors(generatorInputs, NumBlocks, NumThreads);
        assert(error == 0 && "Error during decryptors generation");

        decryptors.read();

        for (auto index = 0; index < decryptors.cpu().size(); ++index, ++gen_seed_global)
        {
            const auto& decryptor = decryptors.cpu()[index];

            const bool match = decryptor.seed() == gen_seed_global;
            assert(match && "Invalid seed bruteforce generation, expected seed does not match generated one");
            if (!match)
            {
                return false;
            }
        }
    }

    return true;
}

bool tests::generators::all()
{
    bool ok = true;
    ok &= pattern();
    ok &= seed();

    return ok;
}

