#include "keeloq_single_result.h"

#include <stdio.h>

#include "device/cuda_common.h"


namespace
{
uint32_t SerialFromOTA(uint64_t ota)
{
    return (misc::rev_bits(ota) >> 32) & 0x0FFFFFFF;
}
}

void SingleResult::LearningsArray::print(const KeeloqLearning::LearningItem& item, uint32_t srl, KeeloqLearning::ResultIndex match) const
{
    const auto resIndex = KeeloqLearning::DecryptedResults::getIndex(item);
    const bool ismatch = match == resIndex;

    printf("[%-8s: %-8s: %-8s] Btn:0x%02X | Serial:0x%08X (0x%08" PRIX32 ") | Counter:0x%04X | %7s |\n",
        KeeloqLearning::Name(item.learning), KeeloqLearning::Name(item.imod), KeeloqLearning::Name(item.amod),
        (data[resIndex] >> 28),         // Button
        (data[resIndex] >> 16) & 0x3ff, // Serial
        srl,                            // Serial (OTA)
        data[resIndex] & 0xFFFF,        // Counter
        (ismatch ? "(MATCH)" : ""));
}

void SingleResult::LearningsArray::print() const
{
    for (auto resIndex = 0; resIndex < KeeloqLearning::InvalidResultIndex; ++resIndex)
    {
        print(KeeloqLearning::DecryptedResults::getByIndex(resIndex), -1, KeeloqLearning::NoMatch);
    }
}

void SingleResult::print(const std::vector<EncParcel>& inputs) const
{
    printf("Results (Input: 0x%" PRIX64 " - Man key: 0x%" PRIX64 " - Seed: %u )\n\n",
        inputs[inputIndex].ota, decryptor.man(), decryptor.seed());

    printf("-----------------------------------------------------------------------------------------------------\n");
    for (auto iLearning = 0; iLearning < KeeloqLearning::LearningTypesCount; ++iLearning)
    {
        for (auto iMod = 0; iMod < KeeloqLearning::Modifier::InputModCount; ++iMod)
        {
            for (auto aMod = 0; aMod < KeeloqLearning::Modifier::AlgoModCount; ++aMod)
            {
                const auto l = static_cast<KeeloqLearning::LearningType>(iLearning);
                const auto imod = static_cast<KeeloqLearning::Modifier::Input>(iMod);
                const auto amod = static_cast<KeeloqLearning::Modifier::Algo>(aMod);

                const auto item = KeeloqLearning::LearningItem(l, imod, amod);

                if (KeeloqLearning::DecryptedResults::isValid(item))
                {
                    decrypted.print(item, inputs[inputIndex].srl(), match);
                }
            }
        }
        printf("-----------------------------------------------------------------------------------------------------\n");
    }

    printf("\n");
}