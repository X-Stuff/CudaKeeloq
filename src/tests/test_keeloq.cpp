#include "test_keeloq.h"

#include "device/cuda_vector.h"
#include "algorithm/keeloq/keeloq_kernel.h"

#include "bruteforce/generators/generator_bruteforce.h"


static constexpr auto debug_key = "test_keeloq"_u64;
static constexpr auto debug_seed = 158500;

std::vector<EncParcel> tests::keeloq::gen_inputs(uint64_t key, uint8_t num /*= 3*/, KeeloqLearning::LearningType LType /*= KeeloqLearning::LearningType::Simple*/, KeeloqLearning::Modifier::Type LMod /*= KeeloqLearning::Modifier::Type::Regular*/)
{
    static constexpr uint32_t kDefaultSeed = 987654321;
    static constexpr uint8_t kDefaultButton = 0x3;
    static constexpr uint16_t kDefaultCounter = 0x123;
    static constexpr uint32_t kDefaultSerial = 0xDEADBEEF;

    std::vector<EncParcel> result;
    Encryptor encryptor(key, kDefaultSeed, kDefaultSerial, kDefaultButton, kDefaultCounter);

    for (uint8_t i = 0; i < num; ++i)
    {
        result.emplace_back(encryptor.click(LType, LMod));
    }

    return result;
}

bool tests::keeloq::encparcel()
{
    constexpr uint64_t OTA = 0x1122334455667788;

    constexpr uint32_t FIX = misc::rev_bits(OTA) >> 32;
    constexpr uint32_t HOP = static_cast<uint32_t>(misc::rev_bits(OTA));

    EncParcel parcelOTA(OTA);
    assert(parcelOTA.ota == OTA);
    assert(parcelOTA.fix() == FIX);
    assert(parcelOTA.hop() == HOP);

    EncParcel parcelHopFix(FIX, HOP);
    assert(parcelHopFix.fix() == FIX);
    assert(parcelHopFix.hop() == HOP);
    assert(parcelHopFix.ota == OTA);

    assert(parcelOTA.srl() == parcelHopFix.srl());
    assert(parcelOTA.btn() == parcelHopFix.btn());

    return memcmp(&parcelOTA, &parcelHopFix, sizeof(EncParcel)) == 0;
}


bool tests::keeloq::every_learning_with_mod(const BruteforceConfig& config)
{
    using namespace KeeloqLearning;

    static constexpr auto NumBlocks = 4;
    static constexpr auto NumThreads = 4;

    const bool config_valid = config.brute_size() <= 1 && config.dict_size() <= 1;
    if (!config_valid)
    {
        assertf(config_valid, "For test mode we need just 1 correct run, otherwise benchmark() need to be used!, <brute_size: %lld, dict_size: %lld>",
            config.brute_size(), config.dict_size());

        return false;
    }

    for (auto lMod = 0; lMod < LearningTypesCount; ++lMod)
    {
        for (auto lType = 0; lType < Modifier::Count; ++lType)
        {
            const auto learningType = static_cast<LearningType>(lType);
            const auto learningMod = static_cast<Modifier::Type>(lMod);

            const auto resIndex = DecryptedResults::getIndex(learningType, learningMod);
            if (resIndex == DecryptedResults::InvalidIndex)
            {
                // This combination of learning type and modifier is not valid, skipping
                continue;
            }

            for (auto numInputs = 1; numInputs < 3; ++numInputs)
            {
                Encryptor encryptor(debug_key, debug_seed);

                CudaVector<EncParcel> encrypted = gen_inputs(encryptor, numInputs, learningType, learningMod);
                CudaVector<Decryptor> decryptors(NumBlocks * NumThreads);
                CudaVector<SingleResult> results(decryptors.size() * encrypted.size());

                KeeloqKernelInput generatorInputs;
                generatorInputs.encdata = encrypted.gpu();
                generatorInputs.decryptors = decryptors.gpu();
                generatorInputs.results = results.gpu();
                generatorInputs.Initialize(config, KeeloqLearning::Matrix::Everything());

                if (config.type != BruteforceType::Dictionary)
                {
                    GeneratorBruteforce::PrepareDecryptors(generatorInputs, NumBlocks, NumThreads);
                }
                else
                {
                    generatorInputs.WriteDecryptors(config.decryptors, 0, config.decryptors.size());
                }

                auto result = ::keeloq::kernels::cuda_brute(generatorInputs, NumBlocks, NumThreads);

                // Check were not errors
                if (result.error != 0)
                {
                    assertf(result.error == 0, "Error during bruteforce! Inputs:%d, learning type: %s, modifier: %s",
                        numInputs, KeeloqLearning::Name(learningType), KeeloqLearning::Name(learningMod));
                    return false;
                }

                // Check that we have matches
                if (result.value == 0)
                {
                    assertf(result.value, "Bruteforce didn't succedded! Inputs: %d, learning type: %s, modifier: %s",
                        numInputs, KeeloqLearning::Name(learningType), KeeloqLearning::Name(learningMod));
                    return false;
                }

                // Check that decryptor was correct
                decryptors.read();

                // Should be always first
                const auto& matched_decryptor = decryptors.cpu()[0];
                if (matched_decryptor.man() != debug_key)
                {
                    assertf(matched_decryptor.man() == debug_key,
                        "First decryptor didn't have correct MAN key. Expected: 0x%016llX, got: 0x%016llX. Inputs: %d, learning type: %s, modifier: %s",
                        debug_key, matched_decryptor.man(), numInputs, KeeloqLearning::Name(learningType), KeeloqLearning::Name(learningMod));

                    return false;
                }


                // Check result is correct as well
                results.read();

                // Checking always last
                const auto lastInput = numInputs - 1;
                const auto& matched_result = results.cpu()[lastInput];
                if (matched_result.match != resIndex)
                {
                    assertf(matched_result.match == resIndex,
                        "Result Index didn't match expected index. Expected: %d, got: %d. Num Inputs: %d, learning type: %s, modifier: %s",
                        resIndex, matched_result.match, numInputs, KeeloqLearning::Name(learningType), KeeloqLearning::Name(learningMod));

                    return false;
                }

                // Since click() method increase counter by 1, we need to decrease it back to get correct unencrypted value for comparison
                encryptor.setCounter(encryptor.getCounter() - 1);

                if (matched_result.decrypted[resIndex] != encryptor.unencrypted())
                {
                    assertf(matched_result.decrypted[resIndex] == encryptor.unencrypted(),
                        "Decrypted value at Index: %d, didn't match expected unencrypted value. Expected: 0x%08X, got: 0x%08X. Inputs: %d, learning type: %s, modifier: %s",
                        resIndex, encryptor.unencrypted(), matched_result.decrypted[resIndex], numInputs, KeeloqLearning::Name(learningType), KeeloqLearning::Name(learningMod));

                    return false;
                }
            }
        }
    }

    return true;
}


bool tests::keeloq::every_brute_type()
{
    BruteforceConfig dict = BruteforceConfig::GetDictionary({ Decryptor::Make(debug_key, 0, true) });

    BruteforceConfig brute_n_seed = BruteforceConfig::GetBruteforce(Decryptor::MakeNoSeed(debug_key), 1);
    BruteforceConfig brute_w_seed = BruteforceConfig::GetBruteforce(Decryptor::Make(debug_key, debug_seed, true), 1);

    BruteforceConfig seed = BruteforceConfig::GetSeedBruteforce(Decryptor::Make(debug_key, debug_seed, true));


    std::vector<std::vector<uint8_t>> debug_key_pattern =
    {
        { debug_key >> 56 }, { (debug_key >> 48) & 0xFF }, { (debug_key >> 40) & 0xFF }, { (debug_key >> 32) & 0xFF }, { (debug_key >> 24) & 0xFF }, { (debug_key >> 16) & 0xFF }, { (debug_key >> 8) & 0xFF }, { debug_key & 0xFF }
    };
    BruteforceConfig pattern = BruteforceConfig::GetPattern(Decryptor::MakeNoSeed(debug_key), BruteforcePattern(std::move(debug_key_pattern)), 1);


    std::vector<uint8_t> debug_key_alphabet = { debug_key >> 56, (debug_key >> 48) & 0xFF, (debug_key >> 40) & 0xFF, (debug_key >> 32) & 0xFF, (debug_key >> 24) & 0xFF, (debug_key >> 16) & 0xFF, (debug_key >> 8) & 0xFF, debug_key & 0xFF };
    BruteforceConfig alphabet = BruteforceConfig::GetAlphabet(Decryptor::MakeNoSeed(debug_key), MultibaseDigit(debug_key_alphabet), 1);

    auto all_configs = { dict, brute_n_seed, brute_w_seed, seed, pattern, alphabet };
    for (const auto& config : all_configs)
    {
        printf("--- Testing: %s ", config.toString().c_str());

        if (!every_learning_with_mod(config))
        {
            return false;
        }

        printf("[OK] --- \n");
    }

    return true;
}

bool tests::keeloq::all()
{
    bool ok = true;
    ok &= encparcel();
    ok &= every_brute_type();

    return ok;
}
