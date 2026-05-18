#pragma once

#include <string>

#include <common.h>

#include <kernels/input_transform.h>

#include <algorithm/keeloq/keeloq_decryptor.h>
#include <algorithm/keeloq/keeloq_encrypted.h>
#include <algorithm/keeloq/keeloq_learning_types.h>

/**
 *  A data struct representing information about a single bruteforce item result
 * Item is basically single CUDA thread result
 */
struct BruteforceResult
{
    /** Used decryptor */
    Decryptor decryptor = {};

    /** Decrypted value */
    uint32_t decrypted = 0;

    /** One of the used input */
    EncParcel input = {};

    /** Input transform applied during decryption */
    InputsTransform transform = InputsTransform::None;

    /** Learning type used in decryption */
    KeeloqLearning::LearningType learningType = KeeloqLearning::LearningType::Simple;

    /** And additional algorithm modifier used in decryption */
    KeeloqLearning::Modifier::Algo algoModifier = KeeloqLearning::Modifier::Algo::Normal;

public:
    BruteforceResult(bool isMatched, const Decryptor& inDecryptor, uint32_t inDecrypted, const EncParcel& inInput, InputsTransform inTransform, KeeloqLearning::LearningType inLearningType, KeeloqLearning::Modifier::Algo inAlgoModifier)
        : decryptor(inDecryptor), decrypted(inDecrypted), input(inInput), transform(inTransform), learningType(inLearningType), algoModifier(inAlgoModifier), match(isMatched)
    {
    }

private:
    BruteforceResult() = default;

public:
    static BruteforceResult Invalid() { return BruteforceResult(); }

public:
    /** If this result is a valid at all */
    inline __host__ bool isValid() const { return input.ota != 0 && decrypted != 0 && decryptor.is_valid(); }

    /** If this result is a valid match (MAN is successfully restored) */
    __host__ bool isMatch() const;

    /** Serializes self into string and prints to console */
    __host__ void print() const;

    /** Serializes self into string */
    __host__ std::string toString() const;

private:
    /** If this result is a valid match (MAN is successfully restored) */
    bool match = false;
};
