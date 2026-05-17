#include "bruteforce/bruteforce_result.h"

bool BruteforceResult::isMatch() const
{
    assert((!match || ((decrypted >> 28) == input.btn() && ((decrypted >> 16) & 0x3ff) == input.srl())) &&
        "Match flag is true but decrypted data doesn't match the unencrypted input's part");

    return isValid() && match;
}

void BruteforceResult::print() const
{
    printf("%s\n", toString().c_str());
}

std::string BruteforceResult::toString() const
{
    return str::format<std::string>(
        "[%-8s: %-8s: %-8s] | Man Key: 0x%" PRIX64 " (Seed: %u) | Btn:0x%02X | Serial:0x%08X (0x%03" PRIX32 ") | Counter:0x%04X | %7s |",
        KeeloqLearning::name(algoModifier), KeeloqLearning::name(learningType), name(mutation),
        decryptor.man(), decryptor.seed(),

        (decrypted >> 28),          // Button
        input.serial(),             // Serial (OTA 28 bits)
        (decrypted >> 16) & 0x3ff,  // Serial (10 bits)
        decrypted & 0xFFFF,         // Counter
        (isMatch() ? "(MATCH)" : "")
    );
}
