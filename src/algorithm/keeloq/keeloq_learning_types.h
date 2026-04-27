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

    /************************************************************************/

    Explicit = 1 << 0,

    Force = 1 << 1,

    /************************************************************************/

    // Type: Normal learning type (without seed)
    Normal = 1 << 2,

    // Type: Learning types with seed also
    Seeded = 1 << 3,

    /************************************************************************/

    // Modifier: Disable Regular decryption
    NoReg = 1 << 4,

    // Modifier: Disable Reverse manufacturer key calculations
    NoRev = 1 << 5,

    // Modifier: Disable Inverted algorithms calculations (for Normal, Secure, FAAC learning types)
    NoInv = 1 << 6,

    ModMask = NoReg | NoRev | NoInv,

    /************************************************************************/

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

__host__ __device__ __inline__ constexpr LearningDecryptionMode operator&(const LearningDecryptionMode& a, const LearningDecryptionMode& b)
{
    return static_cast<LearningDecryptionMode>(static_cast<int>(a) & static_cast<int>(b));
}

__host__ __device__ __inline__ constexpr LearningDecryptionMode operator|(const LearningDecryptionMode& a, const LearningDecryptionMode& b)
{
    return static_cast<LearningDecryptionMode>(static_cast<int>(a) | static_cast<int>(b));
}

__host__ __device__ __inline__ constexpr bool operator!(LearningDecryptionMode m)
{
    return m == static_cast<LearningDecryptionMode>(0);
}

namespace KeeloqLearning
{

/**
 * reference: https://github.com/DarkFlippers/unleashed-firmware/blob/dev/lib/subghz/protocols/keeloq_common.h
 */
enum LearningType : uint8_t
{
    // Simple Learning
    Simple = 0,

    //###########################
    // Normal Learning
    // https://phreakerclub.com/forum/showpost.php?p=43557&postcount=37
    Normal,

    Secure,

    // Magic xor type1 learning
    Xor,

    Faac,

    Serial1,

    Serial2,

    Serial3,
};

template<LearningType... Values> struct LearningTypesSet
{
    static constexpr CudaFixedArray<LearningType, sizeof...(Values)> values = { Values... };

    constexpr auto begin() const { return values[0]; }

    constexpr auto end() const { return values[values.size() - 1]; }
};

using SeededTypes = LearningTypesSet<LearningType::Secure, LearningType::Faac>;

using NormalTypes = LearningTypesSet<LearningType::Simple, LearningType::Normal, LearningType::Xor, LearningType::Serial1, LearningType::Serial2, LearningType::Serial3>;

/** Number of learning types */
static constexpr const uint8_t LearningTypesCount = static_cast<uint8_t>(LearningType::Serial3) + 1;

/**
 * Helper alias, defines sequence of all available learning types indices (0, 1, 2, ..., LearningTypesCount-1)
 */
using LearningTypesSequence = std::make_index_sequence<LearningTypesCount>;

/**
 *  Additional modification for learning types, in some cases might be useful
 */
struct Modifier
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
    static constexpr uint8_t Count = static_cast<uint8_t>(Type::InvertedDec) + 1;

    using TypeSequence = std::make_index_sequence<Count>;
};

__host__ __device__ __inline__ constexpr Modifier::Mask operator|(const Modifier::Mask& a, const Modifier::Mask& b) { return static_cast<Modifier::Mask>(static_cast<uint8_t>(a) | static_cast<uint8_t>(b)); }
__host__ __device__ __inline__ constexpr Modifier::Mask operator&(const Modifier::Mask& a, const Modifier::Mask& b) { return static_cast<Modifier::Mask>(static_cast<uint8_t>(a) & static_cast<uint8_t>(b)); }
__host__ __device__ __inline__ constexpr bool operator!(Modifier::Mask m) { return m == static_cast<Modifier::Mask>(0); }


/** Helper alias for reverse lookup */
struct Pair
{
    LearningType type;

    Modifier::Type mod;

    __host__ __device__ __forceinline__ static constexpr uint8_t asIndex(LearningType type, Modifier::Type mod)
    {
        return static_cast<uint8_t>(type) * Modifier::Count + static_cast<uint8_t>(mod);
    }
};

/**
 *  Meta-programming helpers struct for definition of Indices and ReverseIndices arrays in KeeloqLearningMatrix.
 */
struct Registry
{
    /**
     *  Internal information for each learning type that will be in Available tuple.
     */
    template<LearningType LType, bool bIsSeeded, Modifier::Type... LMods>
    struct Entry
    {
        static constexpr LearningType Type = LType;

        static constexpr Modifier::Mask ModsMask = (static_cast<Modifier::Mask>(0) | ... | Modifier::ToMask<LMods>());

        /** Cannot detect repeated */
        static constexpr uint8_t NumMods = sizeof...(LMods);
        static_assert(NumMods <= Modifier::Count, "Too many modifications, cannot be more than known types!");

        static constexpr bool IsSeeded = bIsSeeded;

        template<Modifier::Type M>
        __host__ __device__ __inline__ static constexpr bool HasMod() { return !!(ModsMask & Modifier::ToMask<M>()) != 0; }

        __host__ __device__ __inline__ static constexpr bool HasMod(Modifier::Type M) { return !!(ModsMask & Modifier::ToMask(M)) != 0; }

        /** Index in Mods collection */
        template<Modifier::Type M>
        __host__ __device__ __inline__ static constexpr std::enable_if_t<HasMod<M>(), uint8_t> ModIndex()
        {
            uint8_t index = 0;
            bool found = false;

            // Fold expression: increment 'index' for every element
            // until 'found' becomes true (when LMods == M)
            ((found || (LMods == M ? (found = true, false) : (index++, false))), ...);

            return index;
        }

        static constexpr uint8_t Indices[Modifier::Count] = { 0 };
    };

    /**
     *  All available learning types with their modifications. If you want to add new learning type - just add it here with allowed modifications.
     */
    using Available = std::tuple<
        Entry<LearningType::Simple, false,  Modifier::Type::Regular, Modifier::Type::ReversedKey>,
        Entry<LearningType::Normal, false,  Modifier::Type::Regular, Modifier::Type::ReversedKey, Modifier::Type::InvertedDec>,
        Entry<LearningType::Secure, true,   Modifier::Type::Regular, Modifier::Type::ReversedKey, Modifier::Type::InvertedDec>,
        Entry<LearningType::Xor,    false,  Modifier::Type::Regular, Modifier::Type::ReversedKey>,
        Entry<LearningType::Faac,   true,   Modifier::Type::Regular, Modifier::Type::ReversedKey, Modifier::Type::InvertedDec>,
        Entry<LearningType::Serial1, false, Modifier::Type::Regular, Modifier::Type::ReversedKey>,
        Entry<LearningType::Serial2, false, Modifier::Type::Regular, Modifier::Type::ReversedKey>,
        Entry<LearningType::Serial3, false, Modifier::Type::Regular, Modifier::Type::ReversedKey>
    >;

    /** Element accessor */
    template<std::size_t I> using Element = std::tuple_element_t<I, Available>;

    template<std::size_t... I>
    inline static constexpr uint8_t TotalResults(std::index_sequence<I...>)
    {
        return (Element<I>::NumMods + ...);
    }

    /** Total number of available learning types with allowed modifications */
    static constexpr uint8_t NumResults = TotalResults(LearningTypesSequence{});

private:
    template<std::size_t... I>
    static constexpr bool ValidateOrder(std::index_sequence<I...>)
    {
        static_assert(sizeof...(I) == LearningTypesCount, "Size of the sequence doesn't match learning types number");
        static_assert(((Element<I>::Type == static_cast<LearningType>(I)) && ...), "Incompatible learning type/modifier combination");

        return ((Element<I>::Type == static_cast<LearningType>(I)) && ...);
    }

    static constexpr bool Validate()
    {
        constexpr bool sizeValid = LearningTypesCount == static_cast<uint8_t>(LearningType::Serial3) + 1;
        static_assert(sizeValid, "LearningTypesCount should match KeeloqLearningType count");

        constexpr bool arrayValid = ValidateOrder(LearningTypesSequence{});
        static_assert(arrayValid, "Learning types array elements mismatch");

        return sizeValid && arrayValid;
    }
public:
    Registry()
    {
        static_assert(Validate(), "Learning types validation failed");
        static_assert(LearningTypesCount == std::tuple_size_v<Available>, "AvailableLearnings definition missing some elements");
    }
};

/** Alias for indexer type in DecryptedResults, from index you can easily restore the Learning-Mod pair */
using ResultIndex = uint8_t;

/** Value if the result index that represents no match, Invalid Index points to the last (additional) element in array, however it is considered invalid */
static constexpr ResultIndex NoMatch = 0xFF;

template<LearningType LType, Modifier::Type LMod>
struct IndexInResults
{
    static constexpr auto ModMask = Modifier::ToMask<LMod>();

    template<std::size_t... I>
    static constexpr ResultIndex count(std::index_sequence<I...>)
    {
        using RegElement = typename Registry::Element<LType>;

        if constexpr (RegElement::template HasMod<LMod>())
        {
            constexpr auto BaseIndex = (Registry::Element<I>::NumMods + ... + 0);

            return BaseIndex + RegElement::template ModIndex<LMod>();
        }

        return Registry::NumResults;
    }

    static constexpr uint8_t value = count(std::make_index_sequence<LType>{});
};

/** Size of the full matrix N*M with indices that points to results array */
static constexpr const uint8_t IndicesCacheSize = LearningTypesCount * Modifier::Count;

/**
 *  Indices cache, initialized from ResultIndices:values, static per translation unit in CUDA.
 * DecryptedResults::cuda_init() should be called at startup.
 */
extern __constant__ CudaFixedArray<uint8_t, IndicesCacheSize> IndicesCache;

/** Decoded array type, used to have exact memory fit for all learning types with modifications, and one last element for invalid combination */
struct DecryptedResults : public CudaFixedArray<uint32_t, Registry::NumResults + 1>
{
    /** Index of the last element where points all invalid for specific learning type modifications  */
    static constexpr ResultIndex InvalidIndex = Registry::NumResults;

    template<uint8_t... Is>
    struct ResultIndices
    {
        static constexpr CudaFixedArray<uint8_t, sizeof...(Is)> values = { Is... };

        __host__ __device__ __inline__ static constexpr uint8_t get(uint8_t at)
        {
            return values[at];
        }

        __host__ __device__ __inline__ static constexpr uint8_t get(LearningType learning, Modifier::Type mod)
        {
            return values[Pair::asIndex(learning, mod)];
        }
    };

private:
    /**
     *  Magic template alias that generates array of indices for all learning types and modifications, based on Registry::Available definition.
     */
    template<std::size_t M, std::size_t N, std::size_t... I>
    using TIndicesType = ResultIndices<
        (Registry::Element<((I / N))>::HasMod(static_cast<Modifier::Type>(I% N))
            ? IndexInResults<static_cast<LearningType>((I / N)), static_cast<Modifier::Type>(I% N)>::value
            : InvalidIndex)...>;

    /**
     *  Another magic template struct that helps generate sequence with ALL indices with simple definition through `make_index_sequence`
     */
    template<std::size_t M, std::size_t N, typename Seq>
    struct MakeTIndicesType;

    /**
     *  Partial specialization of MakeTIndicesType for index sequence,
     * generates TIndicesType with all indices for M learning types and N modifications.
     */
    template<std::size_t M, std::size_t N, std::size_t... I>
    struct MakeTIndicesType<M, N, std::index_sequence<I...>>
    {
        using type = TIndicesType<M, N, I...>;
    };

public:
    /**
     *  Magically defined type with indices in results collection for all learning types and modifications, based on Registry::Available definition.
     *
     * Looks like this:
     *
     *      Simple Normal  Secure  Xor  Faac  Serial1 Serial2 Serial3
     * Reg:   0      2       5      8    10     13      15     17
     * Rev:   1      3       6      9    11     14      16     18
     * Inv:   X      4       7      X    12     XX      XX     XX
     *
     * Where X == 19 (last index in DecryptedResults) - means invalid combination of learning type and modification
     *
     */
    using ResultIndicesCache = typename MakeTIndicesType<LearningTypesCount, Modifier::Count, std::make_index_sequence<IndicesCacheSize>>::type;

public:
    /** Get index in DecryptedResults for specific learning type with modification */
    template<LearningType LType, Modifier::Type LMod>
    __host__ __device__ __inline__ static constexpr uint8_t getIndex()
    {
        return IndexInResults<LType, LMod>::value;
    }

    /** Get index in DecryptedResults for specific learning type with modification, not allowed on device! */
    __host__ __device__ __inline__ static constexpr uint8_t getIndex(LearningType type, Modifier::Type mod)
    {
#if __CUDA_ARCH__
        return IndicesCache[Pair::asIndex(type, mod)];
#else
        return ResultIndicesCache::get(type, mod);
#endif

    }

    /** Does reverse lookup by index, not allowed on device! */
    __host__ __inline__ static constexpr const auto getByIndex(ResultIndex index)
    {
        for (auto iLearning = 0; iLearning < LearningTypesCount; ++iLearning)
        {
            for (auto iMod = 0; iMod < Modifier::Count; ++iMod)
            {
                const auto l = static_cast<LearningType>(iLearning);
                const auto m = static_cast<Modifier::Type>(iMod);

                if (ResultIndicesCache::get(l, m) == index)
                {
                    return Pair{ l, m };
                }
            }
        }

        return Pair{};
    }
public:

    __host__ static cudaError_t cuda_init()
    {
        constexpr auto src = DecryptedResults::ResultIndicesCache::values;
        return cudaMemcpyToSymbol(IndicesCache.data, src.data, sizeof(src));
    }
};


/**
 *  Used for:
 *
 *  - getIndex(L, M) - fast indexation to the results array
 *  - Setting up dynamic learning matrix, to select specific learnings and thier modifications
 */
struct Matrix
{
    static constexpr auto kEverything = static_cast<uint64_t>(-1);

    __host__ __device__ __inline__ constexpr Matrix(uint64_t value = 0) : matrix(value)
    {
#if __CUDA_ARCH__
        // In CUDA we do not have static_assert with message
#else
        static_assert(DecryptedResults::getIndex(LearningType::Normal, Modifier::Type::Regular) == 2, "Invalid index for Normal/Regular");
        static_assert(DecryptedResults::getIndex(LearningType::Simple, Modifier::Type::InvertedDec) == DecryptedResults::InvalidIndex, "Simple learning should not have valid index for Inverted decode");

        static_assert(DecryptedResults::getIndex(LearningType::Normal, Modifier::Type::Regular) == DecryptedResults::getIndex<LearningType::Normal, Modifier::Type::Regular>(), "getIndex() methods returned non-equal values");
        static_assert(DecryptedResults::getIndex(LearningType::Simple, Modifier::Type::InvertedDec) == DecryptedResults::getIndex<LearningType::Simple, Modifier::Type::InvertedDec>(), "getIndex() methods returned non-equal values");

        static_assert(Registry::NumResults == DecryptedResults::InvalidIndex, "TotalNum should match InvalidIndex it's basically the same");
        static_assert(LearningTypesCount == std::tuple_size_v<Registry::Available>, "AvailableLearnings definition missing some elements");
#endif
    }

    /** Creates a learning matrix with specific enabled pairs */
    Matrix(const std::initializer_list<Pair>& pairs);

    Matrix(const std::vector<LearningType>& types, Modifier::Mask mods);

    /** Creates a learning matrix with everything enabled */
    __host__ __device__ __inline__ static constexpr auto Everything() { return Matrix(kEverything); }

public:
    /**
     *  If specific bit at index is enabled,
     * If you have for loop up to InvalidIndex - is better to use this method
     */
    __host__ __device__ __inline__ bool isEnabled(uint8_t bitIndex) const
    {
        return (matrix & (1ULL << bitIndex)) != 0;
    }

    /**
     *  Matrix access as double indexation [type][mod]
     */
    __host__ __device__ __inline__ bool isEnabled(LearningType type, Modifier::Type mod) const
    {
        const auto index = DecryptedResults::getIndex(type, mod);
        const bool valid = index != DecryptedResults::InvalidIndex;
        return valid && isEnabled(index);
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
    __host__ __inline__ void enable(LearningType type, Modifier::Type mod = Modifier::Type::Regular)
    {
        const auto bitIndex = DecryptedResults::getIndex(type, mod);
        matrix |= (1ULL << bitIndex);
    }

    __host__ std::string to_string() const;

private:
    //  64 bits are too much, we have only 24 possible combinations,
    // however we use only 19 since some combinations are invalid (e.g. Simple learning with InvertedDec modification).
    //
    // Matrix can look like this:
    //
    //      Simple Normal  Secure  Xor  Faac  Serial1 Serial2 Serial3
    // Reg:   0      0       0      0     0      0       0      1
    // Rev:   1      0       0      0     0      0       0      1
    // Inv:   0      0       0      0     0      0       0      0
    //
    //
    uint64_t matrix;
};

const char* Name(LearningType type);

const char* Name(Modifier::Type mod);
}
