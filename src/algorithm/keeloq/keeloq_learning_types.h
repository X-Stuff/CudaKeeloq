#pragma once

#include "common.h"

#include <vector>
#include <string>
#include <type_traits>
#include <cuda_runtime_api.h>


// Array of learning booleans
// If value at index which equals KeeloqLearningType::Type is true
// that means this learning type is enabled
using KeeloqLearningMask = uint8_t[];

/**
 * reference: https://github.com/DarkFlippers/unleashed-firmware/blob/dev/lib/subghz/protocols/keeloq_common.h
 */
struct KeeloqLearningType
{
    using Type = uint8_t;

    enum : Type
    {
        Simple = 0,
        Simple_Rev,

        Normal,
        Normal_Rev,

        Secure,
        Secure_Rev,

        Xor,
        Xor_Rev,

        Faac,
        Faac_Rev,

        Serial1,
        Serial1_Rev,

        Serial2,
        Serial2_Rev,

        Serial3,
        Serial3_Rev,

        LAST,

        // Name for internal use
        ALL = LAST,

        // For array initialization
        TypeMaskLength = ALL + 1,

        INVALID = 0xff,
    };

public:

    static std::string to_string(const Type learning_types[]);
    static std::string to_string(const std::vector<Type>& learning_types);

    static void full_mask(Type out_mask[]) { to_mask({}, out_mask); }
    static void to_mask(const std::vector<Type>& in_types, Type out_mask[]);

    static constexpr const char* ValueString(Type type)
    {

        constexpr const char* LUT[]{
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "10", "11", "12", "13", "14", "15", "16", "17", "18", "19",
            "20", "21", "22", "23", "24", "25", "26", "27", "28", "29",
            "30", "31", "32"
        };

        return LUT[type];
    }

    static constexpr const char* Name(Type type)
    {
        if (type >= LearningNamesCount) {
            return "INVALID";
        }

        return LearningNames[type];
    }

    // Checks if
    template<Type type>
    __device__ __host__ static inline bool OneEnabled(const KeeloqLearningMask mask)
    {
        static_assert(type <= KeeloqLearningType::LAST, "Invalid learning type provided! It Should be less or equal to LAST. LAST means all");
        return mask[type];
    }

    __device__ __host__ static inline bool AllEnabled(const KeeloqLearningMask mask)
    {
        return OneEnabled<KeeloqLearningType::LAST>(mask);
    }

private:

    static const char* LearningNames[];

    static const size_t LearningNamesCount;
};


/**
 * Simple struct to have nicer code when want to use full mask
 */
struct KeeloqAllLearningsMask
{
    KeeloqLearningType::Type mask[KeeloqLearningType::TypeMaskLength];

    __device__ __host__ KeeloqAllLearningsMask()
    {
        UNROLL
        for (uint8_t i = 0; i < KeeloqLearningType::ALL; ++i)
        {
            mask[i] = 0;
        }
        mask[KeeloqLearningType::ALL] = 1;
    }
};