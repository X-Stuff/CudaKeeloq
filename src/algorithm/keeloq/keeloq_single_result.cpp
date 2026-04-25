#include "keeloq_single_result.h"

#include <stdio.h>

#include "device/cuda_common.h"


namespace
{
    uint32_t SerialFromOTA(uint64_t ota)
    {
        return misc::rev_bits(ota, sizeof(ota) * 8) >> 32 & 0x0FFFFFFF;
    }
}

void SingleResult::DecryptedArray::print(uint8_t resIndex, uint64_t ota, bool ismatch) const
{
    const auto [lrn, mode] = KeeloqLearningMatrix::getByIndex(resIndex);

    printf("[%-40s (%-8s)] Btn:0x%X\tSerial:0x%X (0x%" PRIX32 ")\tCounter:0x%X\t%s\n", KeeloqLearning::Name(lrn), KeeloqLearning::Name(mode),
        (data[resIndex] >> 28),              // Button
        (data[resIndex] >> 16) & 0x3ff,      // Serial
        SerialFromOTA(ota),                  // Serial (OTA)
        data[resIndex] & 0xFFFF,             // Counter
        (ismatch ? "(MATCH)" : ""));
}

void SingleResult::DecryptedArray::print() const
{
    for (uint8_t resIndex = 0; resIndex < KeeloqLearningMatrix::InvalidResultIndex; ++resIndex)
    {
        print(resIndex, -1, false);
    }
}

void SingleResult::print(bool onlymatch /* = true */) const
{
    printf("Results (Input: 0x%" PRIX64 " - Man key: 0x%" PRIX64 " - Seed: %u )\n\n",
        encrypted.ota, decryptor.man(), decryptor.seed());

    for (uint8_t resIndex = 0; resIndex < KeeloqLearningMatrix::InvalidResultIndex; ++resIndex)
    {
        bool isMatch = match == resIndex;
        if (!onlymatch)
        {
            decrypted.print(resIndex, encrypted.ota, isMatch);
        }
        else if (isMatch)
        {
            decrypted.print(resIndex, encrypted.ota, isMatch);
        }
    }
    printf("\n");
}