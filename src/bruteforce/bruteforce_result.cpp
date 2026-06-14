#include "bruteforce/bruteforce_result.h"

#include "algorithm/keeloq/keeloq_thread_result.h"

bool BruteforceResult::isMatch() const
{
    assert((!match || ThreadResult::matchDecoded(learningType, decrypted, input)) &&
        "Match flag is true but decrypted data doesn't match the unencrypted input's part");

    return isValid() && match;
}

void BruteforceResult::print() const
{
    printf("%s\n", toString().c_str());
}

std::string BruteforceResult::toString() const
{
    // FAAC SLH keeps serial+button in the cleartext fixed code (serial<<4|button) with a 20-bit
    // counter in the hop and no 10-bit serial; standard KeeLoq carries button/serial(10)/counter(16)
    // in the decrypted hop.
    const bool isFaac = learningType == KeeloqLearning::LearningType::Faac;

    const uint32_t button   = isFaac ? (input.fix() & 0xF)   : (decrypted >> 28);
    const uint32_t serial28 = isFaac ? (input.fix() >> 4)    : input.serial();
    const uint32_t serial10 = isFaac ? 0u                    : ((decrypted >> 16) & 0x3ff);
    const uint32_t counter  = isFaac ? (decrypted & 0xFFFFF) : (decrypted & 0xFFFF);

    return str::format<std::string>(
        "[%-8s: %-8s: %-8s] | Man Key: 0x%" PRIX64 " (Seed: 0x%" PRIX32 ") | Btn:0x%02X | Serial:0x%08X (0x%03" PRIX32 ") | Counter:0x%04X | %7s |",
        KeeloqLearning::name(algoType), KeeloqLearning::name(learningType), InputTransformName(transform).c_str(),
        decryptor.man(), decryptor.seed(),

        button,    // Button
        serial28,  // Serial (OTA 28 bits)
        serial10,  // Serial (10 bits)
        counter,   // Counter
        (isMatch() ? "(MATCH)" : "")
    );
}
