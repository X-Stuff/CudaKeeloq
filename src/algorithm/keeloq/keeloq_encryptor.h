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

    static constexpr uint32_t kRandomCounter = 0x123;

    static constexpr uint32_t kRandomSerial = 0xDEADBEEF;

public:
    Encryptor(uint64_t key) : Encryptor(key, kRandomSeed, kRandomSerial, kRandomButton, kRandomCounter)
    {
    }

    Encryptor(uint64_t key, uint32_t seed) : Encryptor(key, seed, kRandomSerial, kRandomButton, kRandomCounter)
    {
    }

    Encryptor(uint64_t key, uint32_t seed, uint32_t serial, uint8_t button, uint32_t count) :
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
    EncParcel click(InputsTransform inTransform, KeeloqLearning::LearningType ltype, KeeloqLearning::AlgoType algoType);

    /** Derives the effective manufacturer key for the given learning type and algorithm type. */
    uint64_t man(InputsTransform inTransform, KeeloqLearning::LearningType ltype, KeeloqLearning::AlgoType algoType) const;

    /** Assembled hopping-code value prior to encryption for @ltype (FAAC SLH uses a distinct layout from standard KeeLoq). */
    uint32_t unencrypted(KeeloqLearning::LearningType ltype) const;

    /** Assembled fixed code (button + serial) for @ltype — constant per device (FAAC SLH uses a distinct layout from standard KeeLoq). */
    uint32_t fixed(KeeloqLearning::LearningType ltype) const;

    /** Current counter value. Counter is 16bit, but in some rare cases like FAAC SLH - 20bit */
    inline uint32_t getCounter() const { return count; }

    /** Overrides the counter (for tests that need a deterministic counter). */
    inline void setCounter(uint32_t value) { count = value; }

private:

    /** CPU encryption result — raw, not bit-reversed, not OTA. */
    uint32_t cpuEncrypt(InputsTransform inTransform, KeeloqLearning::LearningType ltype, KeeloqLearning::AlgoType algoType) const;

    /** CPU decryption of an OTA value with the given learning/algorithm type. */
    uint32_t cpuDecrypt(uint64_t enc, InputsTransform inTransform, KeeloqLearning::LearningType ltype, KeeloqLearning::AlgoType algoType) const;

    /** GPU encryption result — raw, not bit-reversed, not OTA. */
    uint32_t gpuEncrypt(InputsTransform inTransform, KeeloqLearning::LearningType ltype, KeeloqLearning::AlgoType algoType) const;

    /** GPU decryption of an OTA value with the given learning/algorithm type. */
    uint32_t gpuDecrypt(uint64_t enc, InputsTransform inTransform, KeeloqLearning::LearningType ltype, KeeloqLearning::AlgoType algoType) const;

private:

    uint64_t key;

    uint32_t seed;

    uint32_t serial;

    uint8_t button;

    uint32_t count;
};
