#include "test_keeloq.h"

#include "algorithm/keeloq/keeloq_kernel.h"


std::vector<EncParcel> tests::keeloq::gen_inputs(uint64_t key, uint8_t num /*= 3*/,
    uint32_t serial /*= 0xDEADBEEF*/, uint16_t counter /*= 0x123*/, uint8_t button /*= 0x3*/, uint32_t seed /*= 987654321*/,
    KeeloqLearningType::Type learning /*= KeeloqLearningType::Normal*/)
{
    std::vector<EncParcel> result { ::keeloq::GetOTA(key, seed, serial, button, counter, learning) };

    for (uint8_t i = 1; i < num; ++i)
    {
        result.emplace_back(::keeloq::GetOTA(key, seed, serial, button, counter + i, learning));
    }

    return result;
}
