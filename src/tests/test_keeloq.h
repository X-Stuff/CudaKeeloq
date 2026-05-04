#pragma once

#include <vector>

#include "algorithm/keeloq/keeloq_encrypted.h"
#include "algorithm/keeloq/keeloq_encryptor.h"
#include "algorithm/keeloq/keeloq_kernel.h"
#include "algorithm/keeloq/keeloq_learning_types.h"


namespace tests
{
namespace keeloq
{

using namespace KeeloqLearning;

/** Generate `num` encrypted parcels using the provided encryptor, learning, and modifiers. */
inline std::vector<EncParcel> genInputs(Encryptor& encryptor, uint8_t num = 3, LearningType LType = LearningType::Simple, Modifier::Input iMod = Modifier::Input::Normal, Modifier::Algo aMod = Modifier::Algo::Normal)
{
    std::vector<EncParcel> result;

    for (uint8_t i = 0; i < num; ++i)
    {
        result.emplace_back(encryptor.click(LType, iMod, aMod));
    }

    return result;
}

/** Generate `num` encrypted parcels for a fresh Encryptor seeded with the given key/learning/modifiers. */
std::vector<EncParcel> genInputs(uint64_t key, uint8_t num = 3, LearningType lType = LearningType::Simple, Modifier::Input iMod = Modifier::Input::Normal, Modifier::Algo aMod = Modifier::Algo::Normal);

/** EncParcel round-trip test (OTA ↔ fix/hop representations). */
bool encparcel();

/** Exhaustive learning × modifier test driven by the supplied BruteforceConfig. */
bool everyLearningWithMod(const BruteforceConfig& config);

/** Run everyLearningWithMod over every supported bruteforce attack type. */
bool everyBruteType();

/** KernelResult accumulator sanity test (match flag + thread count). */
bool checkKernelResults();

/** Launch the full keeloq test suite. */
bool all();

}
}
