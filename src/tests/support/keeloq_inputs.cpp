#include "tests/support/keeloq_inputs.h"


namespace
{
constexpr uint32_t kDefaultSeed    = 987654321;
constexpr uint8_t  kDefaultButton  = 0x3;
constexpr uint16_t kDefaultCounter = 0x123;
constexpr uint32_t kDefaultSerial  = 0xDEADBEEF;
}

std::vector<EncParcel> tests::keeloq::genInputs(uint64_t key, uint8_t num, InputsMutation inputMutation, LearningType lType, Modifier::Algo aMod)
{
    Encryptor encryptor(key, kDefaultSeed, kDefaultSerial, kDefaultButton, kDefaultCounter);
    return genInputs(encryptor, num, inputMutation, lType, aMod);
}
