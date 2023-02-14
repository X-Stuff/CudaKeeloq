#pragma once

#include <vector>

#include "algorithm/keeloq/keeloq_learning_types.h"

namespace tests
{
namespace keeloq
{
    // Generate test inputs for specific key
    std::vector<uint64_t> gen_inputs(uint64_t key,
        uint32_t serial = 0xDEADBEEF, uint16_t counter = 0x123, uint8_t button = 0x3, KeeloqLearningType::Type learning = KeeloqLearningType::Normal);

}
}