#include "keeloq_single_result.h"

#include <stdio.h>

#include "device/cuda_common.h"


namespace
{
uint32_t SerialFromOTA(uint64_t ota)
{
    return misc::rev_bits(ota) >> 32 & 0x0FFFFFFF;
}
}

void SingleResult::LearningsArray::print(uint8_t resIndex, uint64_t ota, bool ismatch) const
{
    const auto [lrn, mode] = KeeloqLearning::DecryptedResults::getByIndex(resIndex);

    printf("[%-8s: %-11s]\tBtn:0x%02X | Serial:0x%08X (0x%08" PRIX32 ") | Counter:0x%04X | %7s |\n", KeeloqLearning::Name(lrn), KeeloqLearning::Name(mode),
        (data[resIndex] >> 28),              // Button
        (data[resIndex] >> 16) & 0x3ff,      // Serial
        SerialFromOTA(ota),                  // Serial (OTA)
        data[resIndex] & 0xFFFF,             // Counter
        (ismatch ? "(MATCH)" : ""));
}

void SingleResult::LearningsArray::print() const
{
    for (auto resIndex = 0; resIndex < KeeloqLearning::DecryptedResults::InvalidIndex; ++resIndex)
    {
        print(resIndex, -1, false);
    }
}

void SingleResult::print(const std::vector<EncParcel>& inputs, bool onlymatch /* = true */) const
{
    printf("Results (Input: 0x%" PRIX64 " - Man key: 0x%" PRIX64 " - Seed: %u )\n\n",
        inputs[inputIndex].ota, decryptor.man(), decryptor.seed());

    for (auto resIndex = 0; resIndex < KeeloqLearning::DecryptedResults::InvalidIndex; ++resIndex)
    {
        const bool isMatch = match == resIndex;
        if (!onlymatch)
        {
            decrypted.print(resIndex, inputs[inputIndex].ota, isMatch);
        }
        else if (isMatch)
        {
            decrypted.print(resIndex, inputs[inputIndex].ota, isMatch);
        }
    }
    printf("\n");
}