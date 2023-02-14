#include "test_keeloq.h"

#include "algorithm/keeloq/keeloq_kernel.h"


std::vector<uint64_t> tests::keeloq::gen_inputs(uint64_t key,
    uint32_t serial /*= 0xDEADBEEF*/, uint16_t counter /*= 0x123*/, uint8_t button /*= 0x3*/, KeeloqLearningType::Type learning /*= KeeloqLearningType::Normal*/)
{
    return {
        ::keeloq::GetOTA(key, serial, button, counter + 0, learning),
        ::keeloq::GetOTA(key, serial, button, counter + 1, learning),
        ::keeloq::GetOTA(key, serial, button, counter + 2, learning),
    };
}
