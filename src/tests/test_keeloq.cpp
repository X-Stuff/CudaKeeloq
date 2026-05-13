#include "doctest/doctest.h"

#include "algorithm/keeloq/keeloq_decryptor.h"
#include "algorithm/keeloq/keeloq_encryptor.h"
#include "algorithm/keeloq/keeloq_encrypted.h"
#include "algorithm/keeloq/keeloq_kernel.h"
#include "algorithm/keeloq/keeloq_learning_types.h"
#include "algorithm/keeloq/keeloq_single_result.h"

#include "bruteforce/bruteforcer.h"
#include "bruteforce/bruteforce_config.h"
#include "bruteforce/bruteforce_pattern.h"
#include "bruteforce/generators/generator_bruteforce.h"

#include "device/cuda_vector.h"

#include "kernels/kernel_result.h"
#include "kernels/kernel_input_multi_learning.h"

#include "tests/support/keeloq_inputs.h"

#include <cstdio>


using namespace KeeloqLearning;

namespace
{
constexpr auto kDebugKey  = "test_keeloq"_u64;
constexpr auto kDebugSeed = 158500;


// Exercises every valid learning  x  input-mod  x  algo-mod combination for a given
// bruteforce config, asserting the decryptor and result match.
//
// Returns early on the first real failure so the CAPTURE() block explains what
// combination failed.
void runEveryLearningWithMod(const BruteforceConfig& config)
{
    static const CudaConfig cudaConfig = CudaConfig::Tests();

    const bool isCorrectSizeForTests = config.bruteSize() == 1 || config.dictSize() == 1;
    REQUIRE_MESSAGE(isCorrectSizeForTests, "config must produce exactly one real run (use benchmark tests otherwise)");

    uint16_t totalMatches = 0;

    for (auto lType = 0; lType < LearningTypesCount; ++lType)
    {
        const auto learningType = static_cast<LearningType>(lType);

        // MESSAGE() would prepend file:line. For simple progress output we want
        // just the label — print to stdout directly.
        std::printf("TESTING LEARNING: %s\n", KeeloqLearning::name(learningType));
        std::fflush(stdout);

        auto inputMutations = config.getMutations();

        for (auto mutation : inputMutations)
        {
            const bool seedRequired = KeeloqLearning::hasSeed(learningType) || !!(mutation & InputsMutation::XorFix);

            const bool canMatch = (config.hasSeed() && KeeloqLearning::hasSeed(learningType)) || (!seedRequired && config.type != BruteforceType::Seed);

            for (auto aMod = 0; aMod < Modifier::AlgoModCount; ++aMod)
            {
                const auto algoModifier   = static_cast<Modifier::Algo>(aMod);
                const auto learningItem = LearningItem(learningType, algoModifier);

                const auto resIndex = DecryptedResults::getIndex(learningItem);
                if (resIndex == KeeloqLearning::InvalidResultIndex)
                {
                    continue;  // invalid combination
                }

                for (auto numInputs = 1; numInputs <= 3; ++numInputs)
                {
                    CAPTURE(std::string(KeeloqLearning::name(learningType)));
                    CAPTURE(std::string(KeeloqLearning::name(algoModifier)));
                    CAPTURE(std::string(name(mutation)));

                    CAPTURE(numInputs);

                    Encryptor encryptor(kDebugKey, kDebugSeed);
                    const auto inputs = tests::keeloq::genInputs(encryptor, numInputs, mutation, learningType, algoModifier);

                    Bruteforcer bruteforcer(inputs, false, AppVerbosity::Error);
                    auto ccopy = config;
                    ccopy.overrideMutationMask(mutation, true);
                    bruteforcer.setOnRoundComplete([&](const BruteforceRound& round, const KernelResult& kernelResult)
                    {
                        auto& kernelInputs = round.inputs();

                        auto decryptors = kernelInputs.decryptors;
                        REQUIRE(decryptors != nullptr);
                        auto copiedDecryptors = decryptors->read();

                        REQUIRE(kernelResult.cudaError == cudaSuccess);
                        REQUIRE(kernelResult.threadsFinished() == cudaConfig.threadsCount());

                        if (canMatch)
                        {
                            REQUIRE(kernelResult.hasMatch());
                            totalMatches++;

                            // First decryptor must carry the MAN key we seeded the test with
                            CHECK(copiedDecryptors[0].man() == kDebugKey);

                            // Last input's result index must match the learning combination
                            const auto matched_result = kernelInputs.getResult(numInputs - 1);
                            CHECK(matched_result.isValid());
                            CHECK(matched_result.isMatch());

                            CHECK(matched_result.mutation == mutation);
                            CHECK(matched_result.learningType == learningType);
                            CHECK(matched_result.algoModifier == algoModifier);

                            // click() advances the counter, roll it back before comparing
                            encryptor.setCounter(encryptor.getCounter() - 1);
                            CHECK(matched_result.decrypted == encryptor.unencrypted());
                        }
                    });
                    bruteforcer.run(ccopy, cudaConfig, KeeloqLearning::Matrix::Everything());
                }
            }
        }
    }

    MESSAGE("Finished. Total matches: ", totalMatches);
}
}  // namespace


TEST_CASE("keeloq: flat")
{
    static const CudaConfig cudaConfig = CudaConfig::Tests();

    static const auto config = BruteforceConfig::GetBruteforce(Decryptor::Make(kDebugKey, kDebugSeed, true), InputsMutation::All, 1);

    uint16_t totalMatches = 0;

    for (auto lType = 0; lType < LearningTypesCount; ++lType)
    {
        const auto learningType = static_cast<LearningType>(lType);

        auto inputMutations = config.getMutations();

        for (auto mutation : inputMutations)
        {
            for (auto aMod = 0; aMod < Modifier::AlgoModCount; ++aMod)
            {
                const auto algoModifier = static_cast<Modifier::Algo>(aMod);
                const auto learningItem = LearningItem(learningType, algoModifier);

                const auto resIndex = DecryptedResults::getIndex(learningItem);

                for (auto numInputs = 1; numInputs <= 3; ++numInputs)
                {
                    Encryptor encryptor(kDebugKey, kDebugSeed);
                    const auto inputs = tests::keeloq::genInputs(encryptor, numInputs, mutation, learningType);

                    CudaVector<Decryptor> decryptors(cudaConfig.total());
                    CudaVector<SingleLearningResult> results(decryptors.size() * inputs.size());

                    KeeloqKernelSingleLearningInput kernelInputs;
                    kernelInputs.decryptors = decryptors.gpu();
                    kernelInputs.results = results.gpu();
                    kernelInputs.Initialize(config, inputs);

                    auto cudaError = GeneratorBruteforce::PrepareDecryptors(kernelInputs, cudaConfig);
                    REQUIRE(cudaError == cudaSuccess);

                    decryptors.read();

                    kernelInputs.BruteforcePrepare(mutation, learningType, algoModifier);
                    keeloq::kernels::cuda_brute(kernelInputs, cudaConfig);
                }
            }
        }
    }
}

TEST_CASE("keeloq: KernelResult accumulates match flag and thread count")
{
    KernelResult kr;

    kr.onKernelFinish(1);
    CHECK(kr.hasMatch());
    CHECK(kr.threadsFinished() == 1);

    kr.onKernelFinish(0);
    CHECK(kr.hasMatch());
    CHECK(kr.threadsFinished() == 2);

    kr.onKernelFinish(1);
    CHECK(kr.hasMatch());
    CHECK(kr.threadsFinished() == 3);
}

TEST_CASE("keeloq: EncParcel round-trips between OTA and fix/hop")
{
    constexpr uint64_t OTA = 0x1122334455667788;
    const uint32_t FIX = misc::rev_bits(OTA) >> 32;
    const uint32_t HOP = static_cast<uint32_t>(misc::rev_bits(OTA));

    EncParcel parcelOTA(OTA);
    CHECK(parcelOTA.ota == OTA);
    CHECK(parcelOTA.fix() == FIX);
    CHECK(parcelOTA.hop() == HOP);

    EncParcel parcelHopFix(FIX, HOP);
    CHECK(parcelHopFix.fix() == FIX);
    CHECK(parcelHopFix.hop() == HOP);
    CHECK(parcelHopFix.ota == OTA);

    CHECK(parcelOTA.srl() == parcelHopFix.srl());
    CHECK(parcelOTA.btn() == parcelHopFix.btn());
    CHECK(memcmp(&parcelOTA, &parcelHopFix, sizeof(EncParcel)) == 0);
}

// Each of the following TEST_CASEs runs a full GPU sweep over every valid
// learning x algo-mod combination for one bruteforce type. They are
// independent so a single failure does not obscure the others; MESSAGE() prints
// the name unconditionally so the reporter shows progress while the sweep runs.

TEST_CASE("keeloq: learning x modifier sweep: Dictionary")
{
    MESSAGE("Dictionary");
    runEveryLearningWithMod(BruteforceConfig::GetDictionary({ Decryptor::Make(kDebugKey, kDebugSeed, true) }, InputsMutation::All));
}

TEST_CASE("keeloq: learning x modifier sweep: Bruteforce (no seed)")
{
    MESSAGE("Bruteforce (no seed)");
    runEveryLearningWithMod(BruteforceConfig::GetBruteforce(Decryptor::MakeNoSeed(kDebugKey), InputsMutation::All, 1));
}

TEST_CASE("keeloq: learning x modifier sweep: Bruteforce (with seed)")
{
    MESSAGE("Bruteforce (with seed)");
    runEveryLearningWithMod(BruteforceConfig::GetBruteforce(Decryptor::Make(kDebugKey, kDebugSeed, true), InputsMutation::All, 1));
}

TEST_CASE("keeloq: learning x modifier sweep: Seed")
{
    MESSAGE("Seed");
    runEveryLearningWithMod(BruteforceConfig::GetSeedBruteforce(Decryptor::Make(kDebugKey, kDebugSeed, true), InputsMutation::All, 1));
}

TEST_CASE("keeloq: learning x modifier sweep: Pattern")
{
    MESSAGE("Pattern");
    std::vector<std::vector<uint8_t>> bytes = {
        { kDebugKey & 0xFF },         { (kDebugKey >> 8)  & 0xFF },
        { (kDebugKey >> 16) & 0xFF }, { (kDebugKey >> 24) & 0xFF },
        { (kDebugKey >> 32) & 0xFF }, { (kDebugKey >> 40) & 0xFF },
        { (kDebugKey >> 48) & 0xFF }, { uint8_t(kDebugKey >> 56) },
    };
    runEveryLearningWithMod(BruteforceConfig::GetPattern(Decryptor::Make(kDebugKey, kDebugSeed, true), InputsMutation::All, BruteforcePattern(std::move(bytes)), 1));
}

TEST_CASE("keeloq: learning x modifier sweep: Alphabet")
{
    MESSAGE("Alphabet");
    std::vector<uint8_t> alphabet = {
        uint8_t(kDebugKey & 0xFF),         uint8_t((kDebugKey >> 8)  & 0xFF),
        uint8_t((kDebugKey >> 16) & 0xFF), uint8_t((kDebugKey >> 24) & 0xFF),
        uint8_t((kDebugKey >> 32) & 0xFF), uint8_t((kDebugKey >> 40) & 0xFF),
        uint8_t((kDebugKey >> 48) & 0xFF), uint8_t(kDebugKey >> 56),
    };
    runEveryLearningWithMod(BruteforceConfig::GetAlphabet(Decryptor::MakeNoSeed(kDebugKey), InputsMutation::All, MultibaseDigit(alphabet), 1));
}
