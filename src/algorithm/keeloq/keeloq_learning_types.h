#pragma once

#include "common.h"
#include "device/cuda_fixed_array.h"

#include <cuda_runtime_api.h>

#include <string>
#include <vector>
#include <tuple>
#include <type_traits>
#include <string_view>

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
enum KeeloqLearningType : uint8_t
{
    Simple = 0,

    Normal,

    Secure,

    Xor,

    Faac,

    Serial1,

    Serial2,

    Serial3,
};

/** Available number */
static constexpr const uint8_t KeeloqLearningTypesNum = static_cast<uint8_t>(KeeloqLearningType::Serial3) + 1;

/**
 *  Additional modification for learning types, in some cases might be useful
 */
enum KeeloqLearningMod : uint8_t
{
    // Learning type is enabled in normal way
    Regular = 0,

    // Leaning type is enabled with reversed manufacturer key
    ReversedKey,

    // Learning type is enabled with inverted enc/dec logic in learning
    InvertedDec,
};

/** Total number of all modifications */
static constexpr const uint8_t KeeloqLearningModsNum = static_cast<uint8_t>(KeeloqLearningMod::InvertedDec) + 1;

/**
 * Helper alias, defines sequence of all available learning types indices (0, 1, 2, ..., KeeloqLearningTypesNum-1)
 */
using KeeloqLearningTypesSequence = std::make_index_sequence<KeeloqLearningTypesNum>;

/** Helper alias for reverse lookup */
struct LearningPair
{
    KeeloqLearningType type;
    KeeloqLearningMod mod;
};

/**
 *  Meta-programming helpers struct for definition of Indices and ReverseIndices arrays in KeeloqLearningMatrix.
 */
struct KeeloqLearnings
{
    /**
     *  Internal information for each learning type that will be in Available tuple.
     */
    template<KeeloqLearningType LType, bool bIsSeeded, KeeloqLearningMod... LMods>
    struct LearningInfo
    {
        static constexpr KeeloqLearningType Type = LType;

        static constexpr CudaFixedArray<KeeloqLearningMod, sizeof...(LMods)> Mods = { LMods... };

        static constexpr uint8_t NumMods = sizeof...(LMods);

        static constexpr bool IsSeeded = bIsSeeded;
    };

    /**
     *  All available learning types with their modifications. If you want to add new learning type - just add it here with allowed modifications.
     */
    using Available = std::tuple<
        LearningInfo<KeeloqLearningType::Simple, false, KeeloqLearningMod::Regular, KeeloqLearningMod::ReversedKey>,
        LearningInfo<KeeloqLearningType::Normal, false, KeeloqLearningMod::Regular, KeeloqLearningMod::ReversedKey, KeeloqLearningMod::InvertedDec>,
        LearningInfo<KeeloqLearningType::Secure, true, KeeloqLearningMod::Regular, KeeloqLearningMod::ReversedKey, KeeloqLearningMod::InvertedDec>,
        LearningInfo<KeeloqLearningType::Xor, false, KeeloqLearningMod::Regular, KeeloqLearningMod::ReversedKey>,
        LearningInfo<KeeloqLearningType::Faac, true, KeeloqLearningMod::Regular, KeeloqLearningMod::ReversedKey, KeeloqLearningMod::InvertedDec>,
        LearningInfo<KeeloqLearningType::Serial1, false, KeeloqLearningMod::Regular, KeeloqLearningMod::ReversedKey>,
        LearningInfo<KeeloqLearningType::Serial2, false, KeeloqLearningMod::Regular, KeeloqLearningMod::ReversedKey>,
        LearningInfo<KeeloqLearningType::Serial3, false, KeeloqLearningMod::Regular, KeeloqLearningMod::ReversedKey>
    >;

    /** Element accessor */
    template<std::size_t I> using Element = std::tuple_element_t<I, Available>;

    template<KeeloqLearningType LType>
    struct Offset
    {
        template<std::size_t... I>
        static constexpr auto count(std::index_sequence<I...>)
        {
            return (Element<I>::NumMods + ... + 0);
        }

        inline static constexpr uint8_t value = count(std::make_index_sequence<LType>{});
    };


    template<std::size_t... I>
    inline static constexpr uint8_t TotalTypes(std::index_sequence<I...>)
    {
        return (Element<I>::NumMods + ...);
    }

    /** Total number of available learning types with allowed modifications */
    static constexpr uint8_t TotalNum = TotalTypes(KeeloqLearningTypesSequence{});

private:
    template<std::size_t... I>
    static constexpr bool validate_order(std::index_sequence<I...>)
    {
        static_assert(sizeof...(I) == KeeloqLearningTypesNum, "Size of the sequence doesn't match learning types number");
        static_assert(((Element<I>::Type == static_cast<KeeloqLearningType>(I)) && ...), "Incompatible learning type/modifier combination");

        return ((Element<I>::Type == static_cast<KeeloqLearningType>(I)) && ...);
    }

    static constexpr bool validate_rev_order()
    {
        return true;
    }

    static constexpr bool validate()
    {
        constexpr bool sizeValid = KeeloqLearningTypesNum == static_cast<uint8_t>(KeeloqLearningType::Serial3) + 1;
        static_assert(sizeValid, "KeeloqLearningTypesNum should match KeeloqLearningType count");

        constexpr bool arrayValid = validate_order(KeeloqLearningTypesSequence{});
        static_assert(arrayValid, "Learning types array elements mismatch");

        return sizeValid && arrayValid;
    }
public:
    KeeloqLearnings()
    {
        static_assert(validate(), "Learning types validation failed");

        static_assert(KeeloqLearningTypesNum == std::tuple_size_v<Available>, "AvailableLearnings definition missing some elements");
    }
};

/** Helper alias for modification indices (prettier code) */
using ModIndices = uint8_t[KeeloqLearningModsNum];

using IndicesMatrix = CudaFixedArray<ModIndices, KeeloqLearningTypesNum>;

using RevIndicesArray = CudaFixedArray<LearningPair, KeeloqLearnings::TotalNum>;

/**
 *  Used for:
 *
 *  - Type for fixed array where caclulation results will be stored
 *  - getIndex(L, M) - fast indexation to the results array
 *  - Setting up dynamic learning matrix, to select specific learnings and thier modifications
 */
struct KeeloqLearningMatrix
{
private:
    template<KeeloqLearningType LType>
    __host__ __device__ __inline__ static constexpr auto MakeModIndices(ModIndices result)
    {
        constexpr auto base = KeeloqLearnings::Offset<LType>::value;

        for (auto i = 0; i < KeeloqLearningModsNum; ++i)
        {
            // Set by default as index to invalid (last element in DecryptedResults) result
            result[i] = KeeloqLearnings::TotalNum;
        }

        for (auto i = 0; i < KeeloqLearnings::Element<LType>::Mods.size(); ++i)
        {
            auto mod = KeeloqLearnings::Element<LType>::Mods[i];

            result[mod] = base + i;
        }
    }

    // Array of arrays (matrix) where is indices per learning for each modification
    //
    //      Simple Normal  Secure  Xor  Faac  Serial1 Serial2 Serial3
    // Reg:   0      2       5      8    10     13      15     17
    // Rev:   1      3       6      9    11     14      16     18
    // Inv:   X      4       7      X    12     XX      XX     XX
    //
    // Where X == 19 (last index in DecryptedResults) - means invalid combination of learning type and modification
    template<std::size_t... I>
    __host__ __device__ __inline__ static constexpr auto MakeIndices(std::index_sequence<I...>)
    {
        IndicesMatrix indices{};

        // Fill indices array
        ((MakeModIndices<static_cast<KeeloqLearningType>(I)>(indices[I])), ...);

        return indices;
    }

    template<std::size_t... I>
    __host__ __device__ __inline__ static constexpr RevIndicesArray MakeRevIndices(std::index_sequence<I...>)
    {
        constexpr auto indices = MakeIndices(KeeloqLearningTypesSequence{});
        RevIndicesArray revIndices{};

        auto fillIndex = [&](uint8_t type)
        {
            for (uint8_t mod = 0; mod < KeeloqLearningModsNum; ++mod)
            {
                const auto index = indices[type][mod];
                if (index < KeeloqLearnings::TotalNum)
                {
                    revIndices[index] = { static_cast<KeeloqLearningType>(type), static_cast<KeeloqLearningMod>(mod) };
                }
            }
        };

        (fillIndex(I), ...);

        return revIndices;
    }

public:
    /** Index of the last element where points all invalid for specific learning type modifications  */
    static constexpr uint8_t InvalidResultIndex = KeeloqLearnings::TotalNum;

    /**  */
    static constexpr auto DecryptedResultsSize = InvalidResultIndex + 1;

    /** Decoded array type, used to have exact memory fit for all learning types with modifications, and one last element for invalid combination */
    using DecryptedResults = CudaFixedArray<uint32_t, DecryptedResultsSize>;

    /** Type alias that represents the index of a matching result, from index you can easily restore the Learning-Mod pair */
    using MatchIndex = uint8_t;

    static constexpr MatchIndex NoMatch = 0xFF;

    /** Get index in DecryptedResults for specific learning type with modification */
    __host__ __device__ __inline__ static constexpr uint8_t getIndex(KeeloqLearningType type, KeeloqLearningMod mod)
    {
        constexpr auto indices = MakeIndices(KeeloqLearningTypesSequence{});

        return indices[type][static_cast<uint8_t>(mod)];
    }

    /** UNSAFE! Does reverse lookup by index */
    __host__ __device__ __inline__ static constexpr const auto& getByIndex(uint8_t index)
    {
        constexpr auto reverseIndices = MakeRevIndices(KeeloqLearningTypesSequence{});

        return reverseIndices[index];
    }

public:

    static constexpr auto kEverything = static_cast<uint64_t>(-1);

    __host__ __device__ __inline__ KeeloqLearningMatrix(uint64_t value = 0) : matrix(value)
    {
#if __CUDA_ARCH__
        // In CUDA we do not have static_assert with message
#else
        static_assert(getIndex(KeeloqLearningType::Normal, KeeloqLearningMod::Regular) == 2, "Invalid index for Normal/Regular");
        static_assert(getIndex(KeeloqLearningType::Simple, KeeloqLearningMod::InvertedDec) == InvalidResultIndex, "Simple learning should not have valid index for Inverted decode");

        static_assert(KeeloqLearnings::TotalNum == InvalidResultIndex, "TotalNum should match InvalidResultIndex it's basically the same");
        static_assert(KeeloqLearningTypesNum == std::tuple_size_v<KeeloqLearnings::Available>, "AvailableLearnings definition missing some elements");
#endif
    }

    KeeloqLearningMatrix(std::vector<LearningPair> pairs);

public:
    /**
     *  Matrix access is always as double indexation [type][mod]
     */
    __host__ __device__ __inline__ bool isEnabled(KeeloqLearningType type, KeeloqLearningMod mod) const
    {
        const auto bitIndex = type + mod * KeeloqLearningTypesNum;
        return (matrix & (1ULL << bitIndex)) != 0;
    }

    /**
     *  Special case by default in most cases. GPU expensive since check all DecryptedResults::size() elements.
     */
    __host__ __device__ __inline__ bool isAllEnabled() const
    {
        return matrix == kEverything;
    }

    /**
     *  Set specific bit to 1 according to learning type and modification
     */
    __host__ __inline__ void enable(KeeloqLearningType type, KeeloqLearningMod mod = KeeloqLearningMod::Regular)
    {
        const auto bitIndex = type + mod * KeeloqLearningTypesNum;
        matrix |= (1ULL << bitIndex);
    }

    __host__ std::string to_string() const;

private:
    // 64 bits are too much, we have only 24 possible combinations:
    //
    // Can look like this:
    //
    //      Simple Normal  Secure  Xor  Faac  Serial1 Serial2 Serial3
    // Reg:   0      0       0      0     0      0       0      1
    // Rev:   1      0       0      0     0      0       0      1
    // Inv:   0      0       0      0     0      0       0      0
    //
    // NOTE:
    //  - Here we do not check validity of combination, Inv to Simple can be set to `1`
    uint64_t matrix;
};

namespace KeeloqLearning
{
const char* Name(KeeloqLearningType type);

const char* Name(KeeloqLearningMod mod);
}
