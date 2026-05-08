#pragma once

#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <vector>

#include <cuda_runtime_api.h>

#include "common.h"

#include "device/cuda_fixed_array.h"

struct BruteforceConfig;

/**
 * Bitmask describing which decryption path the kernel takes and which optional
 * learning-type groups / modifiers are included in a single kernel launch.
 */
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
 * Catalogue of supported keeloq learning algorithms.
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

/** Compile-time set of learning types, used to expand template packs. */
template<LearningType... Values> struct LearningTypesSet
{
    static constexpr CudaFixedArray<LearningType, sizeof...(Values)> values = { Values... };

    constexpr auto begin() const { return &values[0]; }

    constexpr auto end() const { return &values[values.size()]; }
};

/** Number of learning types */
static constexpr const uint8_t LearningTypesCount = static_cast<uint8_t>(LearningType::Serial3) + 1;

/**
 * Helper alias, defines sequence of all available learning types indices (0, 1, 2, ..., LearningTypesCount-1)
 */
using LearningTypesSequence = std::make_index_sequence<LearningTypesCount>;

namespace helpers
{
    template<std::size_t... I>
    using EveryLearningTypeImpl = LearningTypesSet<static_cast<LearningType>(I)...>;

    template<typename Seq>
    struct MakeEveryLearningType;

    template<std::size_t... I>
    struct MakeEveryLearningType<std::index_sequence<I...>>
    {
        using type = EveryLearningTypeImpl<I...>;
    };
}

/** Defined all learning types as set of values */
using EveryLearningType = helpers::MakeEveryLearningType<LearningTypesSequence>::type;

/** Defined all seeded learning types as set of values */
using SeededTypes = LearningTypesSet<LearningType::Secure, LearningType::Faac>;

/** Defined all normal (no seed) learning types as set of values */
using NormalTypes = LearningTypesSet<LearningType::Simple, LearningType::Normal, LearningType::Xor, LearningType::Serial1, LearningType::Serial2, LearningType::Serial3>;

/** True if the given learning type requires a seed. */
constexpr bool hasSeed(LearningType type)
{
    for (uint8_t i = 0; i < SeededTypes::values.size(); ++i)
    {
        if (SeededTypes::values[i] == type)
        {
            return true;
        }
    }
    return false;
}


/**
 * Additional tweaks that can be applied on top of a learning algorithm.
 * Some learning types only support a subset of these modifiers.
 */
struct Modifier
{
    enum class Input : uint8_t
    {
        // No input modification, use regular manufacturer key
        Normal = 0,

        // Reverse manufacturer key, use reversed byte order for manufacturer key in calculations
        ReversedKey,
    };

    enum class Algo : uint8_t
    {
        // Regular enc/dec logic, all learning types
        Normal = 0,

        // Inverted enc/dec logic for learning (for Normal, Secure, FAAC learning types)
        Inverted = 1
    };

    /** Total number of algorithm modifications */
    static constexpr uint8_t AlgoModCount = static_cast<uint8_t>(Algo::Inverted) + 1;

    /** Total number of input modifications */
    static constexpr uint8_t InputModCount = static_cast<uint8_t>(Input::ReversedKey) + 1;

    using TypeSequence = std::make_index_sequence<AlgoModCount>;
};

/** Single cell (learning type + input/algo modifiers) in the learning matrix. */
struct LearningItem
{
    constexpr LearningItem() = default;

    constexpr __device__ __host__ LearningItem(LearningType l) : LearningItem(l, Modifier::Input::Normal, Modifier::Algo::Normal)
    {
    }

    constexpr __device__ __host__ LearningItem(LearningType l, Modifier::Input i) : LearningItem(l, i, Modifier::Algo::Normal)
    {
    }

    constexpr __device__ __host__ LearningItem(LearningType l, Modifier::Algo a) : LearningItem(l, Modifier::Input::Normal, a)
    {
    }

    constexpr __device__ __host__ LearningItem(LearningType l, Modifier::Input i, Modifier::Algo a) : learning(l), imod(i), amod(a)
    {
    }

    __device__ __host__ __inline__ constexpr uint8_t asIndex() const
    {
        constexpr auto ImodRowSelector = LearningTypesCount * Modifier::AlgoModCount;
        return (learning + (static_cast<uint8_t>(imod) * ImodRowSelector) + static_cast<uint8_t>(amod) * LearningTypesCount);
    }

    LearningType learning = LearningType::Simple;

    Modifier::Input imod = Modifier::Input::Normal;

    Modifier::Algo amod = Modifier::Algo::Normal;
};

/**
 * Meta-programming registry for enumerating legal learning-type/modifier combinations.
 * Add a new learning type by extending `Available`.
 */
struct Registry
{
    /** Per-learning descriptor: seeded flag and supported algorithm modifiers. */
    template<LearningType LType, bool bIsSeeded, Modifier::Algo... AMods>
    struct Entry
    {
        static constexpr LearningType Type = LType;

        /** Cannot detect repeated */
        static constexpr uint8_t AlgoMask = (static_cast<uint8_t>(0) | ... | static_cast<uint8_t>(1 << static_cast<uint8_t>(AMods)));

        /** Cannot detect repeated */
        static constexpr uint8_t NumMods = sizeof...(AMods);
        static_assert(NumMods <= Modifier::AlgoModCount, "Too many algorithm, modifications, cannot be more than known types!");

        static constexpr bool IsSeeded = bIsSeeded;

        __host__ __device__ __inline__ static constexpr bool HasMod(Modifier::Algo amod) { return (AlgoMask & (1 << static_cast<uint8_t>(amod))) != 0; }

        template<Modifier::Algo AMod>
        __host__ __device__ __inline__ static constexpr bool HasMod() { return HasMod(AMod); }

        /** Index in Mods collection */
        template<Modifier::Algo AMod>
        __host__ __device__ __inline__ static constexpr std::enable_if_t<HasMod<AMod>(), uint8_t> ModIndex()
        {
            uint8_t index = 0;
            bool found = false;

            // Fold expression: increment 'index' for every element
            // until 'found' becomes true (when AMods == AMod)
            ((found || (AMods == AMod ? (found = true, false) : (index++, false))), ...);

            return index;
        }

        static constexpr uint8_t Indices[Modifier::AlgoModCount] = { 0 };
    };

    /**
     *  All available learning types with their modifications. If you want to add new learning type - just add it here with allowed modifications.
     */
    using Available = std::tuple<
        Entry<LearningType::Simple, false,  Modifier::Algo::Normal>,
        Entry<LearningType::Normal, false,  Modifier::Algo::Normal, Modifier::Algo::Inverted>,
        Entry<LearningType::Secure, true,   Modifier::Algo::Normal, Modifier::Algo::Inverted>,
        Entry<LearningType::Xor,    false,  Modifier::Algo::Normal>,
        Entry<LearningType::Faac,   true,   Modifier::Algo::Normal, Modifier::Algo::Inverted>,
        Entry<LearningType::Serial1, false, Modifier::Algo::Normal>,
        Entry<LearningType::Serial2, false, Modifier::Algo::Normal>,
        Entry<LearningType::Serial3, false, Modifier::Algo::Normal>
    >;

    /** Element accessor */
    template<std::size_t I> using Element = std::tuple_element_t<I, Available>;

    /** Number of all algorithms in registry for all learnings */
    template<std::size_t... I>
    static constexpr uint8_t CountRealAlgos(std::index_sequence<I...>)
    {
        return (Element<I>::NumMods + ... + 0);
    }

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

/**
 *  Clang/GNU compatibility, it doesn't allow to initialize constexpr with constexpr method of the same class
 */
struct RegistryInfo
{
    /** Real size of results array, only possible variation of algorithm modification counts */
    static constexpr uint8_t RealAlgosNum = Registry::CountRealAlgos(LearningTypesSequence{});

    /** Real size of results array, only possible variation of algorithm modification counts */
    static constexpr uint8_t RealResultsNum = Modifier::InputModCount * RealAlgosNum;

public:
    template<std::size_t I, std::size_t LCount>
    static constexpr std::size_t ElementIndex() { return I % LCount; }

    template<std::size_t I, std::size_t LCount>
    static constexpr LearningType LearningFromIndex() { return static_cast<LearningType>(I % LCount); }

    template<std::size_t I, std::size_t LCount, uint8_t ICount, uint8_t ACount>
    static constexpr Modifier::Input IModFromIndex() { return static_cast<Modifier::Input>((I / (LCount * ACount) % ICount)); }

    template<std::size_t I, std::size_t LCount, uint8_t ACount>
    static constexpr Modifier::Algo AModFromIndex() { return static_cast<Modifier::Algo>((I / LCount) % ACount); }

};

/** Alias for indexer type in DecryptedResults, from index you can easily restore the Learning-Mod pair */
using ResultIndex = uint8_t;

/** Size of the indices cache, all possible invariations including impossible */
static constexpr const uint8_t IndicesCacheSize = LearningTypesCount * Modifier::AlgoModCount * Modifier::InputModCount;

/** Size of the decrypted array (reduced only to real) */
static constexpr const uint8_t DecryptedArraySize = RegistryInfo::RealResultsNum;

/** Value that points to the last element in the indices cache (Invalid) */
static constexpr const ResultIndex InvalidResultIndex = RegistryInfo::RealResultsNum;


/** Value if the result index that represents no match, Invalid Index points to the last (additional) element in array, however it is considered invalid */
static constexpr ResultIndex NoMatch = 0xFF;

template<LearningType LType, Modifier::Input IMod, Modifier::Algo AMod>
struct IndexInResults
{
    template<std::size_t... I>
    static constexpr ResultIndex count(std::index_sequence<I...> Sequence)
    {
        using RegElement = typename Registry::Element<LType>;

        if constexpr (RegElement::template HasMod<AMod>())
        {
            constexpr auto BaseIndex = Registry::CountRealAlgos(Sequence); // (Registry::Element<I>::NumMods + ... + 0);

            return (BaseIndex + RegElement::template ModIndex<AMod>()) + (static_cast<uint8_t>(IMod) * RegistryInfo::RealAlgosNum);
        }

        return InvalidResultIndex;
    }

    static constexpr uint8_t value = count(std::make_index_sequence<LType>{});
};

/**
 *  Indices cache, initialized from ResultIndices:values, static per translation unit in CUDA.
 * DecryptedResults::cuda_init() should be called at startup.
 */
extern __constant__ CudaFixedArray<ResultIndex, IndicesCacheSize> IndicesCache;

/**
 * Fixed-size buffer of decrypted/encrypted values, one slot per valid
 * (LearningType × Input-mod × Algo-mod) combination plus one terminal "invalid" slot.
 */
struct DecryptedResults : public CudaFixedArray<uint32_t, DecryptedArraySize>
{
    template<uint8_t... Is>
    struct ResultIndices
    {
        static constexpr CudaFixedArray<ResultIndex, sizeof...(Is)> values = { Is... };

        __host__ __device__ __inline__ static constexpr ResultIndex get(uint8_t at)
        {
            return values[at];
        }
    };

private:

    /**
     *  Magic template alias that generates array of indices for all learning types and modifications, based on Registry::Available definition.
     */
    template<std::size_t LCount, uint8_t IModCount, uint8_t AModCount, std::size_t... I>
    using TIndicesType = ResultIndices<
        (Registry::Element<RegistryInfo::ElementIndex<I, LCount>()>::HasMod(RegistryInfo::AModFromIndex<I, LCount, AModCount>())
            ? IndexInResults<
                RegistryInfo::LearningFromIndex<I, LCount>(),
                RegistryInfo::IModFromIndex<I, LCount, IModCount, AModCount>(),
                RegistryInfo::AModFromIndex<I, LCount, AModCount>()
              >::value
            : InvalidResultIndex)...>;

    /**
     *  Another magic template struct that helps generate sequence with ALL indices with simple definition through `make_index_sequence`
     */
    template<std::size_t LCount, uint8_t IModCount, uint8_t AModCount, typename Seq>
    struct MakeTIndicesType;

    /**
     *  Partial specialization of MakeTIndicesType for index sequence,
     * generates TIndicesType with all indices for M learning types and N modifications.
     */
    template<std::size_t LCount, uint8_t IModCount, uint8_t AModCount, std::size_t... I>
    struct MakeTIndicesType<LCount, IModCount, AModCount, std::index_sequence<I...>>
    {
        using type = TIndicesType<LCount, IModCount, AModCount, I...>;
    };

public:
    /**
     *  Magically defined type with indices in results collection for all learning types and modifications, based on Registry::Available definition.
     *
     * Looks like this:
     *                     _____________________________________________________________________
     *                    | Simple | Normal | Secure | Xor | Faac | Serial1 | Serial2 | Serial3 |
     * ___________________|________|________|________|_____|______|_________|_________|_________|
     * Nrm Key:| Normal:  |    0   |    1   |    3   |   5 |   6  |     8   |    9    |    10   |
     *         | Inverted |   XX   |    2   |    4   |  XX |   7  |    XX   |   XX    |    XX   |
     * --------|----------|--------|--------|--------|-----|------|---------|---------|---------|
     * Rev Key:| Normal:  |   11   |   12   |   14   |  16 |  17  |    19   |   20    |    21   |
     *         | Inverted |   XX   |   13   |   15   |  XX |  18  |    XX   |   XX    |    XX   |
     * ¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯
     *
     * Where X == 22 (last index in DecryptedResults) - means invalid combination of learning type and modification
     *
     */
    using ResultIndicesCache = typename MakeTIndicesType<LearningTypesCount, Modifier::InputModCount, Modifier::AlgoModCount, std::make_index_sequence<IndicesCacheSize>>::type;

public:
    /** Get index in DecryptedResults for specific learning type with modification */
    template<LearningType LType, Modifier::Input IMod, Modifier::Algo AMod>
    __host__ __device__ __inline__ static constexpr uint8_t getIndex()
    {
        return IndexInResults<LType, IMod, AMod>::value;
    }

    /** Get index in DecryptedResults for specific learning type with modification */
    __host__ __device__ __forceinline__ static constexpr ResultIndex getIndex(LearningItem lItem)
    {
#if __CUDA_ARCH__
        return IndicesCache[lItem.asIndex()];
#else
        return ResultIndicesCache::get(lItem.asIndex());
#endif
    }

    /** Get index in DecryptedResults for specific learning type with modification */
    __host__ __device__ __forceinline__ static constexpr ResultIndex getIndex(LearningType type, Modifier::Input imod, Modifier::Algo amod)
    {
        return getIndex(LearningItem(type, imod, amod));
    }

    /** Does reverse lookup by index, not allowed on device! */
    __host__ __inline__ static constexpr const auto getByIndex(ResultIndex index)
    {
        for (auto iLearning = 0; iLearning < LearningTypesCount; ++iLearning)
        {
            for (auto iMod = 0; iMod < Modifier::InputModCount; ++iMod)
            {
                for (auto aMod = 0; aMod < Modifier::AlgoModCount; ++aMod)
                {
                    const auto l = static_cast<LearningType>(iLearning);
                    const auto imod = static_cast<Modifier::Input>(iMod);
                    const auto amod = static_cast<Modifier::Algo>(aMod);

                    const auto item = LearningItem(l, imod, amod);

                    if (ResultIndicesCache::get(item.asIndex()) == index)
                    {
                        return item;
                    }
                }
            }
        }

        return LearningItem{};
    }

    __host__ __device__ __inline__ static constexpr bool isValid(LearningItem item)
    {
        return getIndex(item) != InvalidResultIndex;
    }
public:

    __host__ static cudaError_t cuda_init()
    {
        constexpr auto src = DecryptedResults::ResultIndicesCache::values;
        return cudaMemcpyToSymbol(IndicesCache.data, src.data, sizeof(src));
    }
};

/**
 * Bitmask of enabled learning-type/modifier combinations for a single kernel run.
 * Bit-indexed using DecryptedResults::getIndex().
 */
struct Matrix
{
    static constexpr auto kEverything = static_cast<uint64_t>(-1);

    __host__ __device__ __inline__ constexpr explicit Matrix(uint64_t value = 0) : matrix(value)
    {
#if !__CUDA_ARCH__
        static_assert(DecryptedResults::getIndex(LearningType::Simple, Modifier::Input::Normal, Modifier::Algo::Normal) == 0, "Invalid index for Simple/Normal/Normal");
        static_assert(DecryptedResults::getIndex(LearningType::Simple, Modifier::Input::ReversedKey, Modifier::Algo::Normal) == 11, "Invalid index for Simple/RevKey/Normal");
        static_assert(DecryptedResults::getIndex(LearningType::Simple, Modifier::Input::Normal, Modifier::Algo::Inverted) == InvalidResultIndex, "Simple learning should not have valid index for Inverted decode");
        static_assert(DecryptedResults::getIndex(LearningType::Simple, Modifier::Input::ReversedKey, Modifier::Algo::Inverted) == InvalidResultIndex, "Simple learning should not have valid index for RevKey/Inverted");


        static_assert(DecryptedResults::getIndex(LearningType::Normal, Modifier::Input::Normal, Modifier::Algo::Normal) == 1, "Invalid index for Normal/Normal/Normal");
        static_assert(DecryptedResults::getIndex(LearningType::Normal, Modifier::Input::Normal, Modifier::Algo::Inverted) == 2, "Invalid index for Normal/Normal/Inverted");
        static_assert(DecryptedResults::getIndex(LearningType::Normal, Modifier::Input::ReversedKey, Modifier::Algo::Normal) == 12, "Invalid index for Normal/RevKey/Normal");
        static_assert(DecryptedResults::getIndex(LearningType::Normal, Modifier::Input::ReversedKey, Modifier::Algo::Inverted) == 13, "Invalid index for Normal/RevKey/Inverted");

        static_assert(DecryptedResults::getIndex(LearningType::Secure, Modifier::Input::Normal, Modifier::Algo::Normal) == 3, "Invalid index for Secure/Normal/Normal");
        static_assert(DecryptedResults::getIndex(LearningType::Secure, Modifier::Input::Normal, Modifier::Algo::Inverted) == 4, "Invalid index for Secure/Normal/Inverted");
        static_assert(DecryptedResults::getIndex(LearningType::Secure, Modifier::Input::ReversedKey, Modifier::Algo::Normal) == 14, "Invalid index for Secure/RevKey/Normal");
        static_assert(DecryptedResults::getIndex(LearningType::Secure, Modifier::Input::ReversedKey, Modifier::Algo::Inverted) == 15, "Invalid index for Secure/RevKey/Inverted");

        static_assert(DecryptedResults::getIndex(LearningType::Xor, Modifier::Input::Normal, Modifier::Algo::Normal) == 5, "Invalid index for Xor/Normal/Normal");
        static_assert(DecryptedResults::getIndex(LearningType::Xor, Modifier::Input::ReversedKey, Modifier::Algo::Normal) == 16, "Invalid index for Xor/RevKey/Normal");
        static_assert(DecryptedResults::getIndex(LearningType::Xor, Modifier::Input::Normal, Modifier::Algo::Inverted) == InvalidResultIndex, "Xor learning should not have valid index for Inverted decode");
        static_assert(DecryptedResults::getIndex(LearningType::Xor, Modifier::Input::ReversedKey, Modifier::Algo::Inverted) == InvalidResultIndex, "Xor learning should not have valid index for RevKey/Inverted");

        static_assert(DecryptedResults::getIndex(LearningType::Faac, Modifier::Input::Normal, Modifier::Algo::Normal) == 6, "Invalid index for Faac/Normal/Normal");
        static_assert(DecryptedResults::getIndex(LearningType::Faac, Modifier::Input::ReversedKey, Modifier::Algo::Normal) == 17, "Invalid index for Faac/RevKey/Normal");
        static_assert(DecryptedResults::getIndex(LearningType::Faac, Modifier::Input::Normal, Modifier::Algo::Inverted) == 7, "Invalid index for Faac/Normal/Inverted");
        static_assert(DecryptedResults::getIndex(LearningType::Faac, Modifier::Input::ReversedKey, Modifier::Algo::Inverted) == 18, "Invalid index for Faac/RevKey/Inverted");

        static_assert(DecryptedResults::getIndex(LearningType::Serial1, Modifier::Input::Normal, Modifier::Algo::Normal) == 8, "Invalid index for Serial1/Normal/Normal");
        static_assert(DecryptedResults::getIndex(LearningType::Serial2, Modifier::Input::Normal, Modifier::Algo::Normal) == 9, "Invalid index for Serial2/Normal/Normal");
        static_assert(DecryptedResults::getIndex(LearningType::Serial3, Modifier::Input::Normal, Modifier::Algo::Normal) == 10, "Invalid index for Serial3/Normal/Normal");
        static_assert(DecryptedResults::getIndex(LearningType::Serial1, Modifier::Input::ReversedKey, Modifier::Algo::Normal) == 19, "Invalid index for Serial1/RevKey/Normal");
        static_assert(DecryptedResults::getIndex(LearningType::Serial2, Modifier::Input::ReversedKey, Modifier::Algo::Normal) == 20, "Invalid index for Serial2/RevKey/Normal");
        static_assert(DecryptedResults::getIndex(LearningType::Serial3, Modifier::Input::ReversedKey, Modifier::Algo::Normal) == 21, "Invalid index for Serial3/RevKey/Normal");

        static_assert(DecryptedResults::getIndex(LearningType::Serial1, Modifier::Input::Normal, Modifier::Algo::Inverted) == InvalidResultIndex, "Serial1 learning should not have valid index for Inverted decode");
        static_assert(DecryptedResults::getIndex(LearningType::Serial2, Modifier::Input::Normal, Modifier::Algo::Inverted) == InvalidResultIndex, "Serial2 learning should not have valid index for Inverted decode");
        static_assert(DecryptedResults::getIndex(LearningType::Serial3, Modifier::Input::Normal, Modifier::Algo::Inverted) == InvalidResultIndex, "Serial3 learning should not have valid index for Inverted decode");
        static_assert(DecryptedResults::getIndex(LearningType::Serial1, Modifier::Input::ReversedKey, Modifier::Algo::Inverted) == InvalidResultIndex, "Serial1 learning should not have valid index for RevKey/Inverted");
        static_assert(DecryptedResults::getIndex(LearningType::Serial2, Modifier::Input::ReversedKey, Modifier::Algo::Inverted) == InvalidResultIndex, "Serial2 learning should not have valid index for RevKey/Inverted");
        static_assert(DecryptedResults::getIndex(LearningType::Serial3, Modifier::Input::ReversedKey, Modifier::Algo::Inverted) == InvalidResultIndex, "Serial3 learning should not have valid index for RevKey/Inverted");

        static_assert(
            DecryptedResults::getIndex(LearningType::Normal, Modifier::Input::Normal, Modifier::Algo::Normal) ==
            DecryptedResults::getIndex<LearningType::Normal, Modifier::Input::Normal, Modifier::Algo::Normal>(),
            "getIndex() methods returned non-equal values");

        static_assert(
            DecryptedResults::getIndex(LearningType::Simple, Modifier::Input::Normal, Modifier::Algo::Inverted) ==
            DecryptedResults::getIndex<LearningType::Simple, Modifier::Input::Normal, Modifier::Algo::Inverted>(),
            "getIndex() methods returned non-equal values");

        static_assert(RegistryInfo::RealResultsNum == InvalidResultIndex, "TotalNum should match InvalidResultIndex it's basically the same");
        static_assert(LearningTypesCount == std::tuple_size_v<Registry::Available>, "AvailableLearnings definition missing some elements");
#endif
    }

    /** Creates a learning matrix with specific enabled pairs */
    Matrix(const std::initializer_list<LearningItem>& pairs);

    /** Creates a learning matrix with specific learning types enabled and all modifications */
    Matrix(const std::initializer_list<LearningType>& types) :
        Matrix(types, { Modifier::Input::Normal, Modifier::Input::ReversedKey }, { Modifier::Algo::Normal, Modifier::Algo::Inverted })
    {
    }

    /** Creates a learning matrix with specific learning and modifications */
    __host__ Matrix(const std::vector<LearningType>& types, const std::vector<Modifier::Input>& iMods, const std::vector<Modifier::Algo>& aMods);

    /** Creates a learning matrix with everything enabled */
    __host__ __device__ __inline__ static constexpr auto Everything() { return Matrix(kEverything); }

    /** Creates a learning matrix with all learnings and all input modifications but only with inverted algorithm enabled */
    __host__ static auto Inverted() { return Matrix({}, { Modifier::Input::Normal, Modifier::Input::ReversedKey }, { Modifier::Algo::Inverted }); }

public:
    /**
     *  If specific bit at index is enabled,
     * If you have for loop up to InvalidResultIndex - is better to use this method
     */
    __host__ __device__ __inline__ bool isEnabled(uint8_t bitIndex) const
    {
        return (matrix & (1ULL << bitIndex)) != 0;
    }

    /**
     *  If specific bit at index encode as [Learning][InputMod][AldoMod] is enabled
     */
    template<LearningType LType, Modifier::Input IMod, Modifier::Algo AMod>
    __host__ __device__ __inline__ bool isEnabled() const
    {
        constexpr auto index = DecryptedResults::getIndex<LType, IMod, AMod>();
        if constexpr (index == InvalidResultIndex)
        {
            return false;
        }
        else
        {
            return isEnabled(index);
        }
    }

    /**
     *  If specific bit at index encode as [Learning][InputMod][AldoMod] is enabled
     * Host only version since uses quite expensive getIndex() method, it is better to use template version if you know learning/modification types at compile time
     */
    __host__ __inline__ bool isEnabled(LearningType type, Modifier::Input imod, Modifier::Algo amod) const
    {
        const auto index = DecryptedResults::getIndex(type, imod, amod);
        const bool valid = index != InvalidResultIndex;
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
     *  Check if matrix has at least something
     */
    __host__ __device__ __inline__ bool isValid() const
    {
        return matrix != 0;
    }

    /**
     *  Set specific bit to 1 according to learning type and modification
     */
    __host__ __inline__ void enable(LearningType type, Modifier::Input imod = Modifier::Input::Normal, Modifier::Algo aMod = Modifier::Algo::Normal)
    {
        const auto bitIndex = DecryptedResults::getIndex(type, imod, aMod);
        matrix |= (1ULL << bitIndex);
    }

    /** Human-readable table of enabled learning/modifier entries. */
    __host__ std::string toString(const BruteforceConfig* config = nullptr) const;

private:
    //  64 bits are too much, we have only 24 possible combinations,
    // however we use only 19 since some combinations are invalid (e.g. Simple learning with InvertedDec modification).
    //
    // Matrix can look like this:
    //
    //                    Simple Normal  Secure  Xor  Faac  Serial1 Serial2 Serial3
    // Nrm Key: Normal  :    0      0       0      0     0      0       0      1
    // Nrm Key: Inverted:    1      0       0      0     0      0       0      1
    // Rev Key: Normal  :    0      0       0      0     0      0       0      1
    // Rev Key: Inverted:    0      0       1      0     0      0       0      1
    //
    //
    uint64_t matrix = 0;
};

/** Human-readable name of a learning type. */
const char* name(LearningType type);

/** Human-readable name of an algorithm modifier. */
const char* name(Modifier::Algo amod);

/** Human-readable name of an input modifier. */
const char* name(Modifier::Input imod);

/** Parses a learning-type name (case-insensitive) or its numeric index. */
bool parse(const char* name, LearningType& out);
}
