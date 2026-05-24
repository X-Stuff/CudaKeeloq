#include "doctest/doctest.h"

#include <algorithm>

#include "algorithm/keeloq/keeloq_kernel.h"

#include "kernels/kernel_input_multi_learning.h"

#include "bruteforce/bruteforcer.h"
#include "bruteforce/bruteforce_config.h"
#include "bruteforce/bruteforce_pattern.h"
#include "bruteforce/generators/generator_bruteforce.h"

#include "device/cuda_vector.h"

#include "tests/support/keeloq_inputs.h"


namespace
{
BruteforceConfig GetSingleKeyConfig(uint64_t key, bool rev = true)
{
    uint8_t* pKey = (uint8_t*)&key;

    std::vector<std::vector<uint8_t>> pattern =
    {
        { pKey[7] }, { pKey[6] }, { pKey[5] }, { pKey[4] },
        { pKey[3] }, { pKey[2] }, { pKey[1] }, { pKey[0] },
    };

    if (rev)
    {
        std::reverse(pattern.begin(), pattern.end());
    }

    const auto inputModifier = rev ? InputsTransform::RevKey : InputsTransform::None;

    BruteforcePattern br_pattern(std::move(pattern), "Test");
    return BruteforceConfig::GetPattern(Decryptor::Make(0, 0, true), inputModifier, br_pattern, 0xFFFFFFFF);
}
}

TEST_CASE("generators: pattern produces the expected first decryptor")
{
    using namespace KeeloqLearning;

    const CudaConfig cudaConfig = CudaConfig::Tests();
    const uint64_t   debugKey   = "hello_world"_u64;

    auto inputsTransform = InputsTransform::None;
    auto inputs = tests::keeloq::genInputs(debugKey, inputsTransform, LearningType::Simple);

    BruteforceConfig config = GetSingleKeyConfig(debugKey);
    config.setLearningMatrix(KeeloqLearning::Matrix::Everything());
    REQUIRE(config.type == BruteforceType::Pattern);
    REQUIRE(config.pattern.init(0).number() == debugKey);

    Bruteforcer bruteforcer(inputs, false, AppVerbosity::Error);
    bruteforcer.setOnRoundComplete([cudaConfig, debugKey](const BruteforceRound& round, const KernelResult& kernelResult)
        {
            auto& kernelInputs = round.inputs();

            auto decryptors = kernelInputs.decryptors;
            REQUIRE(decryptors != nullptr);

            auto copiedDecryptors = decryptors->read();

            REQUIRE(kernelResult.cudaError == cudaSuccess);
            REQUIRE(kernelResult.threadsFinished() == cudaConfig.threadsCount());

            CHECK(copiedDecryptors[0].man() == debugKey);
        });
    bruteforcer.run(config, cudaConfig);
}

TEST_CASE("generators: seed produces a contiguous, monotonically increasing sequence")
{
    using namespace KeeloqLearning;

    constexpr auto NumTestRounds = 8;

    const CudaConfig cudaConfig = CudaConfig::Tests();
    const uint64_t   debugKey   = "hello_world"_u64;

    auto inputsTransform = InputsTransform::None;
    const auto inputs = tests::keeloq::genInputs(debugKey, inputsTransform, LearningType::Secure);

    BruteforceConfig config = BruteforceConfig::GetSeedBruteforce(Decryptor::Make(debugKey, 0, true), inputsTransform);
    REQUIRE(config.type == BruteforceType::Seed);

    KeeloqKernelMultiLearningInput generatorInputs;
    generatorInputs.Initialize(config, inputs);
    generatorInputs.AllocateGPU(cudaConfig.total());

    uint64_t gen_seed_global = 0;
    for (int round = 0; round < NumTestRounds; ++round)
    {
        CAPTURE(round);

        if (round != 0)
        {
            generatorInputs.NextDecryptor();
            gen_seed_global -= 1;
        }

        auto cudaError = GeneratorBruteforce::PrepareDecryptors(generatorInputs, cudaConfig);
        REQUIRE(cudaError == cudaSuccess);

        auto decryptors = generatorInputs.decryptors->read();

        for (size_t index = 0; index < decryptors.size(); ++index, ++gen_seed_global)
        {
            REQUIRE(decryptors[index].seed() == gen_seed_global);
        }
    }
}
