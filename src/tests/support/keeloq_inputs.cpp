#include "tests/support/keeloq_inputs.h"

std::vector<EncParcel> tests::keeloq::genInputs(uint64_t key, InputsTransform inputMutation, LearningType lType, Modifier::Algo aMod)
{
    Encryptor encryptor(key);
    return genInputs(encryptor, inputMutation, lType, aMod);
}
