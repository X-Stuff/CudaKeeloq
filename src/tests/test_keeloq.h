#pragma once

#include <vector>

#include "algorithm/keeloq/keeloq_learning_types.h"
#include "algorithm/keeloq/keeloq_encrypted.h"
#include "algorithm/keeloq/keeloq_encryptor.h"
#include "algorithm/keeloq/keeloq_kernel.h"


namespace tests
{
namespace keeloq
{

using namespace KeeloqLearning;

inline std::vector<EncParcel> gen_inputs(Encryptor& encryptor, uint8_t num = 3, LearningType LType = LearningType::Simple, Modifier::Input iMod = Modifier::Input::Normal, Modifier::Algo aMod = Modifier::Algo::Normal)
{
    std::vector<EncParcel> result;

    for (uint8_t i = 0; i < num; ++i)
    {
        result.emplace_back(encryptor.click(LType, iMod, aMod));
    }

    return result;
}

/**
 *  Generates `num` encrypted inputs with provided key, learning type and modifier.
 */
std::vector<EncParcel> gen_inputs(uint64_t key, uint8_t num = 3, LearningType lType = LearningType::Simple, Modifier::Input iMod = Modifier::Input::Normal, Modifier::Algo aMod = Modifier::Algo::Normal);

/**
 *  Test EncParcel class for valid OTA/fix/hop parsing and generation
 */
bool encparcel();

/**
 *  Test every learning type with every modifier to check if encryption and decryption works correctly.
 * Config describes type of bruteforce
 */
bool every_learning_with_mod(const BruteforceConfig& config);

/**
 *  Test every learning type with every modifier with every brute type
 */
bool every_brute_type();

/**
 *  Test kernel results logic works
 */
bool check_kernel_results();

/**
 *  Launch all tests in this file
 */
bool all();

}
}