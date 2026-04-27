#pragma once

#include "cstdint"

#include "keeloq_encrypted.h"
#include "keeloq_learning_types.h"

/**
 *  Convenience struct for holding encryption parameters for key generation in learning algorithms.
 * Only C++, not used in CUDA kernels.
 * Used only for tests
 */
struct Encryptor
{
    static constexpr uint32_t kRandomSeed = 987654321;

    static constexpr uint8_t kRandmonButton = 0x3;

    static constexpr uint16_t kRandmoCounter = 0x123;

    static constexpr uint32_t kRandomSerial = 0xDEADBEEF;

public:
    Encryptor(uint64_t key) : Encryptor(key, kRandomSeed, kRandomSerial, kRandmonButton, kRandmoCounter)
    {
    }

    Encryptor(uint64_t key, uint32_t seed, uint32_t serial, uint8_t button, uint16_t count) :
        key(key), seed(seed), serial(serial), button(button), count(count)
    {
    }

public:
    /** Generates OTA parcel, increases counter like emulate of button click */
    EncParcel click(KeeloqLearning::LearningType ltype, KeeloqLearning::Modifier::Type lmod);

    /** Returns hopping part before encryption */
    inline uint32_t unencrypted() const { return (uint32_t)button << 28 | ((serial & 0x3FF) << 16) | count; }

    /** Returns fixed part (button, serial, always the same) */
    inline uint32_t fixed() const { return (uint32_t)button << 28 | (serial & 0x0FFFFFFF); }

private:

    static bool validate(uint64_t ota, const Encryptor& encryptor, KeeloqLearning::LearningType ltype, KeeloqLearning::Modifier::Type lmod);

private:

    uint64_t key;

    uint32_t seed;

    uint32_t serial;

    uint8_t button;

    uint16_t count;
};