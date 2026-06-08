#pragma once

#include <vector>

#include "algorithm/keeloq/keeloq_encrypted.h"
#include "algorithm/keeloq/keeloq_encryptor.h"
#include "algorithm/keeloq/keeloq_learning_types.h"


namespace tests
{
namespace keeloq
{

using namespace KeeloqLearning;

inline std::vector<EncParcel> genInputs(Encryptor& encryptor, InputsTransform inTransform = InputsTransform::None,
    LearningType LType = LearningType::Simple, AlgoType algoType = AlgoType::Normal)
{
    static constexpr uint8_t NumInputs = 3;

    std::vector<EncParcel> result;
    result.reserve(NumInputs);

    for (uint8_t i = 0; i < NumInputs; ++i)
    {
        result.emplace_back(encryptor.click(inTransform, LType, algoType));
    }

    return result;
}

std::vector<EncParcel> genInputs(uint64_t key, InputsTransform inTransform = InputsTransform::None,
    LearningType lType = LearningType::Simple, AlgoType algoType = AlgoType::Normal);

}
}
