#pragma once

#include <vector>

#include "algorithm/keeloq/keeloq_learning_types.h"
#include "algorithm/keeloq/keeloq_encrypted.h"

namespace tests
{
namespace keeloq
{
    constexpr uint32_t default_seed = 987654321;

    constexpr uint8_t default_button = 0x3;

    constexpr uint16_t default_counter = 0x123;

    constexpr uint32_t default_serial = 0xDEADBEEF;


    // Generate test inputs for specific key
    std::vector<EncParcel> gen_inputs(uint64_t key, uint8_t num = 3,
        uint32_t serial = default_serial, uint16_t counter = default_counter, uint8_t button = default_button, uint32_t seed = default_seed,
        KeeloqLearningType::Type learning = KeeloqLearningType::Normal);

    template<KeeloqLearningType::Type TLearning>
    inline std::vector<EncParcel> gen_inputs(uint64_t key, uint8_t num = 3,
        uint32_t serial = default_serial, uint16_t counter = default_counter, uint8_t button = default_button, uint32_t seed = default_seed)
    {
       return gen_inputs(key, num, serial, counter, button, seed, TLearning);
    }
}
}