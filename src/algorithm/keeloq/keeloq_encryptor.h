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

    Encryptor(uint64_t key, uint32_t seed) : Encryptor(key, seed, kRandomSerial, kRandmonButton, kRandmoCounter)
    {
    }

    Encryptor(uint64_t key, uint32_t seed, uint32_t serial, uint8_t button, uint16_t count) :
        key(key), seed(seed), serial(serial), button(button), count(count)
    {
    }

public:
    /** Generates OTA parcel, increases counter like emulate of button click */
    EncParcel click(KeeloqLearning::LearningType ltype, KeeloqLearning::Modifier::Type lmod);

    /** Returns new MAN key according to learning type and modifier */
    uint64_t man(KeeloqLearning::LearningType ltype, KeeloqLearning::Modifier::Type lmod) const;

    /** Returns hopping part before encryption */
    inline uint32_t unencrypted() const { return (uint32_t)button << 28 | ((serial & 0x3FF) << 16) | count; }

    /** Returns fixed part (button, serial, always the same) */
    inline uint32_t fixed() const { return (uint32_t)button << 28 | (serial & 0x0FFFFFFF); }

    /** Returns current counter value */
    inline uint16_t getCounter() const { return count; }

    /** Overrides current counter value */
    inline void setCounter(uint16_t value) { count = value; }

private:

    static bool validate(uint64_t ota, const Encryptor& encryptor, KeeloqLearning::LearningType ltype, KeeloqLearning::Modifier::Type lmod);

    /** Returns CPU encrypted value, NOT reversed bits, not OTA */
    uint32_t cpu_encrypt(KeeloqLearning::LearningType ltype, KeeloqLearning::Modifier::Type lmod) const;

    /** Decrypts OTA (rev bits) value on CPU with specific learning and modifier */
    uint32_t cpu_decrypt(uint64_t enc, KeeloqLearning::LearningType ltype, KeeloqLearning::Modifier::Type lmod) const;

    /** Return GPU encrypted value, NOT reversed bits, not OTA */
    uint32_t gpu_encrypt(KeeloqLearning::LearningType ltype, KeeloqLearning::Modifier::Type lmod) const;

    /** Decrypts OTA (rev bits) value on GPU with specific learning and modifier */
    uint32_t gpu_decrypt(uint64_t enc, KeeloqLearning::LearningType ltype, KeeloqLearning::Modifier::Type lmod) const;

private:

    uint64_t key;

    uint32_t seed;

    uint32_t serial;

    uint8_t button;

    uint16_t count;
};