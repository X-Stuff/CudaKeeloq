#include "tests/support/keeloq_inputs.h"

std::vector<EncParcel> tests::keeloq::genInputs(uint64_t key, InputsTransform inputMutation, LearningType lType, AlgoType algoType)
{
    Encryptor encryptor(key);
    return genInputs(encryptor, inputMutation, lType, algoType);
}
