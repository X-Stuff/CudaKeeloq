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

inline std::vector<EncParcel> genInputs(Encryptor& encryptor, uint8_t num = 3, InputsTransform inTransform = InputsTransform::None,
    LearningType LType = LearningType::Simple, Modifier::Algo aMod = Modifier::Algo::Normal)
{
    std::vector<EncParcel> result;
    result.reserve(num);

    for (uint8_t i = 0; i < num; ++i)
    {
        result.emplace_back(encryptor.click(inTransform, LType, aMod));
    }

    return result;
}

std::vector<EncParcel> genInputs(uint64_t key, uint8_t num = 3, InputsTransform inTransform = InputsTransform::None,
    LearningType lType = LearningType::Simple, Modifier::Algo aMod = Modifier::Algo::Normal);

}
}
