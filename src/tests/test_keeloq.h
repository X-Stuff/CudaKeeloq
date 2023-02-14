#pragma once

#include <vector>

#include "algorithm/keeloq/keeloq_learning_types.h"
#include "algorithm/keeloq/keeloq_encrypted.h"

namespace tests
{
namespace keeloq
{
    // Generate test inputs for specific key
    std::vector<EncParcel> gen_inputs(uint64_t key, uint8_t num = 3,
        uint32_t serial = 0xDEADBEEF, uint16_t counter = 0x123, uint8_t button = 0x3, KeeloqLearningType::Type learning = KeeloqLearningType::Normal);
}
}