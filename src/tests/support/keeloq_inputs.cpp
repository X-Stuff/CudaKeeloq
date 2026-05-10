#include "tests/support/keeloq_inputs.h"

std::vector<EncParcel> tests::keeloq::genInputs(uint64_t key, uint8_t num, InputsMutation inputMutation, LearningType lType, Modifier::Algo aMod)
{
    Encryptor encryptor(key);
    return genInputs(encryptor, num, inputMutation, lType, aMod);
}
