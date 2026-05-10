#include "algorithm/keeloq/keeloq_single_result.h"

#include <cstdio>

#include "device/cuda_common.h"

void SingleResult::LearningsArray::print(const KeeloqLearning::LearningItem& item, uint32_t serial, KeeloqLearning::ResultIndex match, InputsMutation mutation) const
{
    const auto resIndex = KeeloqLearning::DecryptedResults::getIndex(item);
    const bool ismatch = match == resIndex;

    printf("[%-8s: %-8s: %-8s] Btn:0x%02X | Serial:0x%08X (0x%08" PRIX32 ") | Counter:0x%04X | %7s |\n",
        KeeloqLearning::name(item.amod), KeeloqLearning::name(item.learning), name(mutation),
        (data[resIndex] >> 28),         // Button
        serial,                         // Serial (OTA 28 bits)
        (data[resIndex] >> 16) & 0x3ff, // Serial (10 bits)
        data[resIndex] & 0xFFFF,        // Counter
        (ismatch ? "(MATCH)" : ""));
}

void SingleResult::print(const std::vector<EncParcel>& inputs) const
{
    printf("Results (Input: 0x%" PRIX64 " (%s) - Man key: 0x%" PRIX64 " - Seed: %u )\n\n",
        inputs[inputIndex()].ota, name(inputsMutation()), decryptor.man(), decryptor.seed());

    printf("-----------------------------------------------------------------------------------------------------\n");
    for (auto aMod = 0; aMod < KeeloqLearning::Modifier::AlgoModCount; ++aMod)
    {
        for (auto iLearning = 0; iLearning < KeeloqLearning::LearningTypesCount; ++iLearning)
        {
            const auto l = static_cast<KeeloqLearning::LearningType>(iLearning);
            const auto amod = static_cast<KeeloqLearning::Modifier::Algo>(aMod);

            const auto item = KeeloqLearning::LearningItem(l, amod);

            if (KeeloqLearning::DecryptedResults::isValid(item))
            {
                decrypted.print(item, inputs[inputIndex()].serial(), match, inputsMutation());
            }
        }
    }
    printf("-----------------------------------------------------------------------------------------------------\n");

    printf("\n");
}
