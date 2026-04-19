#pragma once

#include "common.h"

#include <vector>
#include <string>
#include <type_traits>
#include <cuda_runtime_api.h>


enum class LearningDecryptionMode
{
    // Explicit defined learning types
    Invalid  = 0,

    Explicit = 1 << 0,

    Force = 1 << 1,

    Normal = 1 << 2,

    Seeded = 1 << 3,

    // Disable Reverse manufacturer key calculations
    NoRev = 1 << 4,

    // Run only learning types without seed
    ForceNormal = Force | Normal,

    // Run only learning types with seed
    ForceSeeded = Force | Seeded,

    // Explicit defined but without seed
    ExplicitNormal = Explicit | Normal,

    // Explicit defined but with seed only
    ExplicitSeeded = Explicit | Seeded,

    // RUNS ALL LEARNING TYPES. Seeded Included, even if seed is 0
    ForceAll = ForceNormal | ForceSeeded,

    // Runs runtime checks if learning type need to be calculated (specified via mask)
    ExplicitAll = ExplicitNormal | ExplicitSeeded
};

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

        INVALID = 0xff,
    };

    struct Mask
    {
        friend struct KeeloqLearningType;

        // Default mask when all learning types are enabled
        static constexpr Type All[LAST] = { true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true };

        __host__ __device__ __inline__ bool operator[](uint8_t index) const { return values[index]; }

        void set(uint8_t index, bool is_enabled) { values[index] = is_enabled; }

        bool is_all_enabled() const;

        std::string to_string() const;

        private:
            uint8_t values[LAST] = { 0 };
    };

public:

    static std::string to_string(const std::vector<Type>& learning_types);

    static Mask to_mask(const std::vector<Type>& in_types);
    static Mask full_mask() { return to_mask({}); }

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

private:

    static const char* LearningNames[];

    static const size_t LearningNamesCount;
};
