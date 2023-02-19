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

void SingleResult::DecryptedArray::print(uint8_t element, uint64_t ota, bool ismatch) const
{
    printf("[%-40s] Btn:0x%X\tSerial:0x%X (0x%" PRIX32 ")\tCounter:0x%X\t%s\n", KeeloqLearningType::Name(element),
        (data[element] >> 28),              // Button
        (data[element] >> 16) & 0x3ff,      // Serial
        SerialFromOTA(ota),                 // Serial (OTA)
        data[element] & 0xFFFF,             // Counter
        (ismatch ? "(MATCH)" : ""));
}

void SingleResult::DecryptedArray::print() const
{
    for (uint8_t i = 0; i < ResultsCount; ++i)
    {
        print(i, -1, false);
    }
}

void SingleResult::print(bool onlymatch /* = true */) const
{
    printf("Results (Input: 0x%" PRIX64 " - Man key: 0x%" PRIX64 " )\n\n", encrypted.ota, decryptor.man());

    for (uint8_t i = 0; i < ResultsCount; ++i)
    {
        bool isMatch = match == i;
        if (!onlymatch)
        {
            decrypted.print(i, encrypted.ota, isMatch);
        }
        else if (isMatch)
        {
            decrypted.print(i, encrypted.ota, isMatch);
        }
    }
    printf("\n");
}