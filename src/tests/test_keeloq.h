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

constexpr uint32_t default_seed = 987654321;

constexpr uint8_t default_button = 0x3;

constexpr uint16_t default_counter = 0x123;

constexpr uint32_t default_serial = 0xDEADBEEF;


template<KeeloqLearning::LearningType LType, KeeloqLearning::Modifier::Type LMod = KeeloqLearning::Modifier::Type::Regular>
inline std::vector<EncParcel> gen_inputs(uint64_t key, uint8_t num = 3,
    uint32_t serial = default_serial, uint16_t counter = default_counter, uint8_t button = default_button, uint32_t seed = default_seed)
{
    std::vector<EncParcel> result;
    Encryptor encryptor(key, seed, serial, button, counter);

    for (uint8_t i = 0; i < num; ++i)
    {
        result.emplace_back(encryptor.click(LType, LMod));
    }

    return result;
}

}
}