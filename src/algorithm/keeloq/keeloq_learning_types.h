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

namespace KeeloqLearning
{

/**
 * reference: https://github.com/DarkFlippers/unleashed-firmware/blob/dev/lib/subghz/protocols/keeloq_common.h
 */
enum Type : uint8_t
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
static constexpr const uint8_t TypesNum = static_cast<uint8_t>(Type::Serial3) + 1;

/**
 *  Additional modification for learning types, in some cases might be useful
 */
struct Mod
{
    enum class Type : uint8_t
    {
        // Learning type is enabled in normal way
        Regular = 0,

        // Learning type is enabled with reversed manufacturer key
        ReversedKey = 1,

        // Learning type is enabled with inverted enc/dec logic in learning
        InvertedDec = 2,
    };

    enum class Mask : uint8_t
    {
        Regular = 1 << static_cast<uint8_t>(Type::Regular),

        RevKey = 1 << static_cast<uint8_t>(Type::ReversedKey),

        InvDec = 1 << static_cast<uint8_t>(Type::InvertedDec),

        All = InvDec | Regular | InvDec
    };

    template<Type t>
    __host__ __device__ __inline__ static constexpr Mask ToMask() { return static_cast<Mask>(1 << static_cast<uint8_t>(t)); }

    __host__ __device__ __inline__ static constexpr Mask ToMask(Type t) { return static_cast<Mask>(1 << static_cast<uint8_t>(t)); }

    /** Total number of all modifications */
    static constexpr uint8_t NumTypes = static_cast<uint8_t>(Type::InvertedDec) + 1;

    using TypeSequence = std::make_index_sequence<NumTypes>;
};

__host__ __device__ __inline__ constexpr Mod::Mask operator|(const Mod::Mask& a, const Mod::Mask& b) { return static_cast<Mod::Mask>(static_cast<uint8_t>(a) | static_cast<uint8_t>(b)); }
__host__ __device__ __inline__ constexpr Mod::Mask operator&(const Mod::Mask& a, const Mod::Mask& b) { return static_cast<Mod::Mask>(static_cast<uint8_t>(a) & static_cast<uint8_t>(b)); }
__host__ __device__ __inline__ constexpr bool operator!(Mod::Mask m) { return m == static_cast<Mod::Mask>(0); }

/**
 * Helper alias, defines sequence of all available learning types indices (0, 1, 2, ..., TypesNum-1)
 */
using KeeloqLearningTypesSequence = std::make_index_sequence<TypesNum>;

/** Helper alias for reverse lookup */
struct Pair
{
    Type type;

    Mod::Type mod;
};

/**
 *  Meta-programming helpers struct for definition of Indices and ReverseIndices arrays in KeeloqLearningMatrix.
 */
struct Registry
{
    /**
     *  Internal information for each learning type that will be in Available tuple.
     */
    template<Type LType, bool bIsSeeded, Mod::Type... LMods>
    struct Entry
    {
        static constexpr Type Type = LType;

        static constexpr Mod::Mask ModsMask = (static_cast<Mod::Mask>(0) | ... | Mod::ToMask<LMods>());

        /** Cannot detect repeated */
        static constexpr uint8_t NumMods = sizeof...(LMods);

        static constexpr bool IsSeeded = bIsSeeded;

        template<Mod::Type M>
        __host__ __device__ __inline__ static constexpr bool HasMod() { return !!(ModsMask & Mod::ToMask<M>()) != 0; }

        /** Index in Mods collection */
        template<Mod::Type M>
        __host__ __device__ __inline__ static constexpr std::enable_if_t<HasMod<M>(), uint8_t> ModIndex()
        {
            uint8_t index = 0;
            bool found = false;

            // Fold expression: increment 'index' for every element
            // until 'found' becomes true (when LMods == M)
            ((found || (LMods == M ? (found = true, false) : (index++, false))), ...);

            return index;
        }
    };

    /**
     *  All available learning types with their modifications. If you want to add new learning type - just add it here with allowed modifications.
     */
    using Available = std::tuple<
        Entry<Type::Simple, false,  Mod::Type::Regular, Mod::Type::ReversedKey>,
        Entry<Type::Normal, false,  Mod::Type::Regular, Mod::Type::ReversedKey, Mod::Type::InvertedDec>,
        Entry<Type::Secure, true,   Mod::Type::Regular, Mod::Type::ReversedKey, Mod::Type::InvertedDec>,
        Entry<Type::Xor,    false,  Mod::Type::Regular, Mod::Type::ReversedKey>,
        Entry<Type::Faac,   true,   Mod::Type::Regular, Mod::Type::ReversedKey, Mod::Type::InvertedDec>,
        Entry<Type::Serial1, false, Mod::Type::Regular, Mod::Type::ReversedKey>,
        Entry<Type::Serial2, false, Mod::Type::Regular, Mod::Type::ReversedKey>,
        Entry<Type::Serial3, false, Mod::Type::Regular, Mod::Type::ReversedKey>
    >;

    /** Element accessor */
    template<std::size_t I> using Element = std::tuple_element_t<I, Available>;

    template<std::size_t... I>
    inline static constexpr uint8_t TotalResults(std::index_sequence<I...>)
    {
        return (Element<I>::NumMods + ...);
    }

    /** Total number of available learning types with allowed modifications */
    static constexpr uint8_t NumResults = TotalResults(KeeloqLearningTypesSequence{});

private:
    template<std::size_t... I>
    static constexpr bool validate_order(std::index_sequence<I...>)
    {
        static_assert(sizeof...(I) == TypesNum, "Size of the sequence doesn't match learning types number");
        static_assert(((Element<I>::Type == static_cast<Type>(I)) && ...), "Incompatible learning type/modifier combination");

        return ((Element<I>::Type == static_cast<Type>(I)) && ...);
    }

    static constexpr bool validate()
    {
        constexpr bool sizeValid = TypesNum == static_cast<uint8_t>(Type::Serial3) + 1;
        static_assert(sizeValid, "TypesNum should match KeeloqLearningType count");

        constexpr bool arrayValid = validate_order(KeeloqLearningTypesSequence{});
        static_assert(arrayValid, "Learning types array elements mismatch");

        return sizeValid && arrayValid;
    }
public:
    Registry()
    {
        static_assert(validate(), "Learning types validation failed");
        static_assert(TypesNum == std::tuple_size_v<Available>, "AvailableLearnings definition missing some elements");
    }
};

/** Alias for indexer type in DecryptedResults, from index you can easily restore the Learning-Mod pair */
using ResultIndex = uint8_t;

/** Decoded array type, used to have exact memory fit for all learning types with modifications, and one last element for invalid combination */
using DecryptedResults = CudaFixedArray<uint32_t, Registry::NumResults + 1>;

/** Index of the last element where points all invalid for specific learning type modifications  */
static constexpr ResultIndex InvalidResultIndex = Registry::NumResults;

/** Value if the result index that represents no match, Invalid Index points to the last (additional) element in array, however it is considered invalid */
static constexpr ResultIndex NoMatch = 0xFF;

template<Type LType>
struct BaseIndexInResults
{
    template<std::size_t... I>
    static constexpr auto count(std::index_sequence<I...>)
    {
        return (Registry::Element<I>::NumMods + ... + 0);
    }

    inline static constexpr uint8_t value = count(std::make_index_sequence<LType>{});
};

template<Type LType, Mod::Type LMod>
struct IndexInResults
{
    static constexpr auto ModMask = Mod::ToMask<LMod>();

    template<std::size_t... I>
    static constexpr ResultIndex count(std::index_sequence<I...>)
    {
        constexpr auto BaseIndex = (Registry::Element<I>::NumMods + ... + 0);

        using RegElement = typename Registry::Element<LType>;

        if constexpr (RegElement::template HasMod<LMod>())
        {
            return BaseIndex + RegElement::template ModIndex<LMod>();
        }

        return InvalidResultIndex;
    }

    inline static constexpr uint8_t value = count(std::make_index_sequence<LType>{});
};


/**
 *  Used for:
 *
 *  - getIndex(L, M) - fast indexation to the results array
 *  - Setting up dynamic learning matrix, to select specific learnings and thier modifications
 */
struct Matrix
{
    /** Helper alias for modification indices (prettier code) */
    using ModIndices = uint8_t[Mod::NumTypes];

    using IndicesMatrix = CudaFixedArray<ModIndices, TypesNum>;

    using RevIndicesArray = CudaFixedArray<Pair, Registry::NumResults>;

private:
    template<Type LType, std::size_t... Is>
    __host__ __device__ __inline__ static constexpr auto MakeModIndices(ModIndices result, std::index_sequence<Is...>)
    {
        constexpr auto base = BaseIndexInResults<LType>::value;

        using RegElement = typename Registry::Element<LType>;

        uint8_t index = 0;
        ((result[Is] = (static_cast<uint8_t>(RegElement::ModsMask & Mod::ToMask<static_cast<Mod::Type>(Is)>()) != 0) ? base + (index++) : InvalidResultIndex), ...);
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
        ((MakeModIndices<static_cast<Type>(I)>(indices[I], Mod::TypeSequence{})), ...);

        return indices;
    }

    template<std::size_t... I>
    __host__ __device__ __inline__ static constexpr RevIndicesArray MakeRevIndices(std::index_sequence<I...>)
    {
        constexpr auto indices = MakeIndices(KeeloqLearningTypesSequence{});
        RevIndicesArray revIndices{};

        auto fillIndex = [&](uint8_t type)
        {
            for (uint8_t mod = 0; mod < Mod::NumTypes; ++mod)
            {
                const auto index = indices[type][mod];
                if (index < Registry::NumResults)
                {
                    revIndices[index] = { static_cast<Type>(type), static_cast<Mod::Type>(mod) };
                }
            }
        };

        (fillIndex(I), ...);

        return revIndices;
    }

public:
    /** Get index in DecryptedResults for specific learning type with modification */
    template<Type LType, Mod::Type LMod>
    __host__ __device__ __inline__ static constexpr uint8_t getIndex()
    {
        return IndexInResults<LType, LMod>::value;
    }

    /** Get index in DecryptedResults for specific learning type with modification */
    __host__ __device__ __inline__ static constexpr uint8_t getIndex(Type type, Mod::Type mod)
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

    __host__ __device__ __inline__ constexpr Matrix(uint64_t value = 0) : matrix(value)
    {
#if __CUDA_ARCH__
        // In CUDA we do not have static_assert with message
#else
        static_assert(getIndex(Type::Normal, Mod::Type::Regular) == 2, "Invalid index for Normal/Regular");
        static_assert(getIndex(Type::Simple, Mod::Type::InvertedDec) == InvalidResultIndex, "Simple learning should not have valid index for Inverted decode");

        static_assert(getIndex(Type::Normal, Mod::Type::Regular) == getIndex<Type::Normal, Mod::Type::Regular>(), "getIndex() methods returned non-equal values");
        static_assert(getIndex(Type::Simple, Mod::Type::InvertedDec) == getIndex<Type::Simple, Mod::Type::InvertedDec>(), "getIndex() methods returned non-equal values");

        static_assert(Registry::NumResults == InvalidResultIndex, "TotalNum should match InvalidResultIndex it's basically the same");
        static_assert(TypesNum == std::tuple_size_v<Registry::Available>, "AvailableLearnings definition missing some elements");
#endif
    }

    /** Creates a learning matrix with specific enabled pairs */
    Matrix(const std::initializer_list<Pair>& pairs);

    Matrix(const std::vector<Type>& types, Mod::Mask mods);

    /** Creates a learning matrix with everything enabled */
    static constexpr auto Everything() { return Matrix(kEverything); }

public:
    /**
     *  If specific bit at index is enabled,
     * If you have for loop up to InvalidIndex - is better to use this method
     */
    __host__ __device__ __inline__ bool isEnabled(uint8_t bitIndex) const { return (matrix & (1ULL << bitIndex)) != 0; }

    /**
     *  Matrix access as double indexation [type][mod]
     */
    __host__ __device__ __inline__ bool isEnabled(Type type, Mod::Type mod) const { return isEnabled(type + static_cast<uint8_t>(mod) * TypesNum); }

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
    __host__ __inline__ void enable(Type type, Mod::Type mod = Mod::Type::Regular)
    {
        const auto bitIndex = type + static_cast<uint8_t>(mod) * TypesNum;
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

const char* Name(Type type);

const char* Name(Mod::Type mod);
}
