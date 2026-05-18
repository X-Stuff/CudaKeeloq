#pragma once

#include <cstdint>

#include "algorithm/keeloq/keeloq_encrypted.h"
#include "algorithm/keeloq/keeloq_learning_types.h"
#include "kernels/input_transform.h"


/**
 * Host-only helper that produces keeloq OTA parcels from a known key/seed/serial.
 * Used by the test and bench paths to generate matching ciphertext; never invoked from kernels.
 */
struct Encryptor
{
    static constexpr uint32_t kRandomSeed = 987654321;

    static constexpr uint8_t kRandomButton = 0x3;

    static constexpr uint16_t kRandomCounter = 0x123;

    static constexpr uint32_t kRandomSerial = 0xDEADBEEF;

public:
    Encryptor(uint64_t key) : Encryptor(key, kRandomSeed, kRandomSerial, kRandomButton, kRandomCounter)
    {
    }

    Encryptor(uint64_t key, uint32_t seed) : Encryptor(key, seed, kRandomSerial, kRandomButton, kRandomCounter)
    {
    }

private:
    Encryptor(uint64_t key, uint32_t seed, uint32_t serial, uint8_t button, uint16_t count) :
        key(key), seed(seed), serial(serial), button(button), count(count)
    {
    }

public:
    /** Access key used in this encryptor. Used for demo and tests to actually find the match for inputs of this generator */
    inline uint64_t getKey() const { return key; }

    /** Access seed used in this encryptor. Used for demo and tests to actually find the match for inputs of this generator */
    inline uint32_t getSeed() const { return seed; }

public:
    /** Generates an OTA parcel and bumps the counter (simulates a button click). */
    EncParcel click(InputsTransform inTransform, KeeloqLearning::LearningType ltype, KeeloqLearning::Modifier::Algo amod);

    /** Derives the effective manufacturer key for the given learning type and modifiers. */
    uint64_t man(InputsTransform inTransform, KeeloqLearning::LearningType ltype, KeeloqLearning::Modifier::Algo amod) const;

    /** Assembled hopping-code value prior to encryption. */
    inline uint32_t unencrypted() const { return (uint32_t)button << 28 | ((serial & 0x3FF) << 16) | count; }

    /** Assembled fixed code (button + serial) — constant per device. */
    inline uint32_t fixed() const { return (uint32_t)button << 28 | (serial & 0x0FFFFFFF); }

    /** Current counter value. */
    inline uint16_t getCounter() const { return count; }

    /** Overrides the counter (for tests that need a deterministic counter). */
    inline void setCounter(uint16_t value) { count = value; }

private:

    /** CPU encryption result — raw, not bit-reversed, not OTA. */
    uint32_t cpuEncrypt(InputsTransform inTransform, KeeloqLearning::LearningType ltype, KeeloqLearning::Modifier::Algo amod) const;

    /** CPU decryption of an OTA value with the given learning/modifier. */
    uint32_t cpuDecrypt(uint64_t enc, InputsTransform inTransform, KeeloqLearning::LearningType ltype, KeeloqLearning::Modifier::Algo amod) const;

    /** GPU encryption result — raw, not bit-reversed, not OTA. */
    uint32_t gpuEncrypt(InputsTransform inTransform, KeeloqLearning::LearningType ltype, KeeloqLearning::Modifier::Algo amod) const;

    /** GPU decryption of an OTA value with the given learning/modifier. */
    uint32_t gpuDecrypt(uint64_t enc, InputsTransform inTransform, KeeloqLearning::LearningType ltype, KeeloqLearning::Modifier::Algo amod) const;

private:

    uint64_t key;

    uint32_t seed;

    uint32_t serial;

    uint8_t button;

    uint16_t count;
};
