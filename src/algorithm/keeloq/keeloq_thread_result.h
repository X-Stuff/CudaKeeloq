#pragma once

#include <cstdint>
#include <vector>

#include "common.h"

#include "device/cuda_object.h"

#include "algorithm/keeloq/keeloq_decryptor.h"
#include "algorithm/keeloq/keeloq_encrypted.h"
#include "algorithm/keeloq/keeloq_learning_types.h"
#include "kernels/input_transform.h"


/**
 * Contains result types produced by individual CUDA threads during keeloq bruteforce.
 * Two variants exist to match the two kernel execution strategies:
 *  - Multi:  each thread decrypts with ALL learning types, storing results for every combination.
 *  - Single: each thread decrypts with ONE learning type, controlled from the CPU side.
 */
namespace ThreadResult
{

/**
 * Static helpers for unpacking a 32-bit keeloq decrypted value into its logical fields.
 * Layout: [31:28] button | [27:26] serial (2 bits) not for all | [25:16] serial(10-bit) | [15:0] counter
 */
struct Decode
{
    /** Extract the serial portion (10 bits at [25:16]). */
    __host__ __device__ __forceinline__ static uint32_t serial(uint32_t value)
    {
        return (value >> 16) & 0x3ff;
    }

    /** Extract the button portion (top 4 bits at [31:28]). */
    __host__ __device__ __forceinline__ static uint8_t button(uint32_t value)
    {
        return (value >> 28);
    }

    /** Extract the counter portion (lower 16 bits). */
    __host__ __device__ __forceinline__ static uint16_t counter(uint32_t value)
    {
        return (value & 0x0000FFFF);
    }
};


/**
 * Learning-type-aware validation of a decrypted hop against its parcel's cleartext fixed code,
 * plus the counter used for cross-input consistency. The learning type is a template parameter
 * because every kernel that needs this knows it at compile time, so the branch is resolved away.
 *
 * Standard KeeLoq packs serial+button into the hop, so a match is a direct serial/button equality
 * and the counter is the low 16 bits. FAAC SLH uses a different hop layout and rule (see below).
 */
template<KeeloqLearning::LearningType TLearning>
struct Match
{
    /** True if @decrypted is a valid hop for @enc's fixed code under this learning type. */
    __host__ __device__ __forceinline__ static bool check(uint32_t decrypted, const EncParcel& enc)
    {
        return Decode::serial(decrypted) == enc.srl() && Decode::button(decrypted) == enc.btn();
    }

    /** Counter value used to compare consecutive captures (low 16 bits for standard KeeLoq). */
    __host__ __device__ __forceinline__ static uint32_t counter(uint32_t decrypted)
    {
        return Decode::counter(decrypted);
    }
};

/**
 * FAAC SLH: the decrypted hop is [31:20] = three fixed-code nibbles selected by the decrypted
 * counter parity, [19:0] = 20-bit counter. The serial/button are not in the hop (they live in the
 * cleartext fixed code as `serial << 4 | button`), so a match means the hop's top 12 bits
 * reconstruct those fix nibbles. Mirrors Encryptor::unencrypted(Faac) / Flipper's faac_slh gen_data.
 */
template<>
struct Match<KeeloqLearning::LearningType::Faac>
{
    /** Reconstruct the parity-selected fix nibbles and compare against the hop's top 12 bits. */
    __host__ __device__ __forceinline__ static bool check(uint32_t decrypted, const EncParcel& enc)
    {
        const uint32_t fix = enc.fix();
        const uint32_t top12 = decrypted >> 20;

        // The decrypted counter parity selects which three fixed-code nibbles the top 12 bits must
        // reconstruct. Branchless select (parity is 0/1) avoids a warp-divergent branch.
        const uint32_t parity = counter(decrypted) & 1u;
        const uint32_t even = (nibble(fix, 6) << 8) | (nibble(fix, 7) << 4) | nibble(fix, 5);
        const uint32_t odd  = (nibble(fix, 2) << 8) | (nibble(fix, 3) << 4) | nibble(fix, 4);
        const uint32_t expected = parity * odd + (1u - parity) * even;

        return top12 == expected;
    }

    /** FAAC counter is 20-bit (low 20 bits of the decrypted hop). */
    __host__ __device__ __forceinline__ static uint32_t counter(uint32_t decrypted)
    {
        return decrypted & 0xFFFFFu;
    }

private:
    /** Nibble @i of the fixed code, most-significant first (nibble 0 = bits 28..31, nibble 7 = bits 0..3). */
    __host__ __device__ __forceinline__ static uint32_t nibble(uint32_t fix, uint32_t i)
    {
        return (fix >> (28u - i * 4u)) & 0xFu;
    }
};

/** Runtime dispatch of Match<LType>::check for host code that only knows the learning type at runtime. */
__host__ __forceinline__ bool matchDecoded(KeeloqLearning::LearningType ltype, uint32_t decrypted, const EncParcel& enc)
{
    if (ltype == KeeloqLearning::LearningType::Faac)
    {
        return Match<KeeloqLearning::LearningType::Faac>::check(decrypted, enc);
    }

    // Every non-FAAC learning type uses the primary template's standard serial/button check.
    return Match<KeeloqLearning::LearningType::Simple>::check(decrypted, enc);
}


/**
 * Packed byte that stores per-result metadata: which captured input was used and what transform was applied.
 * Encoding differs between Multi and Single modes due to Single needing a match flag in the same byte.
 */
struct InputInfo
{
    /**
     * Multi-mode packing: [7:4] = inputTransform, [3:0] = inputIndex.
     * No match flag needed here because Multi stores match separately.
     */
    struct Multi
    {
        /** Pack input index and transform into a single byte. */
        __host__ __device__ __forceinline__ static uint8_t pack(uint8_t inputIndex, InputsTransform transform)
        {
            return (static_cast<uint8_t>(transform) << 4) | (inputIndex & 0x0F);
        }

        /** Extract input index [3:0]. */
        __host__ __device__ __forceinline__ static uint8_t getInputIndex(uint8_t packed)
        {
            return packed & 0x0F;
        }

        /** Extract input transform [7:4]. */
        __host__ __device__ __forceinline__ static InputsTransform getTransform(uint8_t packed)
        {
            return static_cast<InputsTransform>(packed >> 4);
        }
    };

    /**
     * Single-mode packing: [7:4] = inputTransform, [3:1] = inputIndex, [0] = hasMatch.
     * The match flag is stored inline because Single mode has no separate match field.
     */
    struct Single
    {
        /** Pack all fields into a single byte. */
        __host__ __device__ __forceinline__ static uint8_t pack(uint8_t hasMatch, uint8_t inputIndex, InputsTransform transform)
        {
            return (static_cast<uint8_t>(transform) << 4) | ((inputIndex << 1) & 0x0E) | (hasMatch & 0x01);
        }

        /** Extract match flag [0]. */
        __host__ __device__ __forceinline__ static bool hasMatch(uint8_t packed)
        {
            return (packed & 0x01) != 0;
        }

        /** Extract input index [3:1]. */
        __host__ __device__ __forceinline__ static uint8_t getInputIndex(uint8_t packed)
        {
            return (packed >> 1) & 0x07;
        }

        /** Extract input transform [7:4]. */
        __host__ __device__ __forceinline__ static InputsTransform getTransform(uint8_t packed)
        {
            return static_cast<InputsTransform>(packed >> 4);
        }
    };
};


/**
 * Per-learning decrypted values array, indexed by `DecryptedResults::getIndex(learning, algoType)`.
 * Provides CUDA-optimized cached reads and field extraction helpers.
 */
struct LearningsArray
{
    /** Fixed-size array for every valid learning/algorithm-type combination. */
    KeeloqLearning::DecryptedResults data;

    /** Cached read of decrypted value at given result index. Uses `__ldca` on device for L1 cache hint. */
    __host__ __device__ __forceinline__ uint32_t operator[](uint8_t index) const
    {
        assert(index < KeeloqLearning::DecryptedResults::Size && "Invalid index of decrypted data. Bigger than last element");
#if __CUDA_ARCH__
        return __ldca(&data[index]);
#else
        return data[index];
#endif
    }

public:
    /** Extract the serial portion of a decrypted entry (10 bits). */
    __host__ __device__ __forceinline__ uint32_t srl(uint8_t index) const
    {
        return Decode::serial((*this)[index]);
    }

    /** Extract the button portion of a decrypted entry (top 4 bits). */
    __host__ __device__ __forceinline__ uint16_t btn(uint8_t index) const
    {
        return Decode::button((*this)[index]);
    }

    /** Extract the counter portion of a decrypted entry (lower 16 bits). */
    __host__ __device__ __forceinline__ uint16_t cnt(uint8_t index) const
    {
        return Decode::counter((*this)[index]);
    }
};


/**
 * Result of a single CUDA thread in multi-learning mode.
 * Stores decrypted values for ALL learning/algorithm-type combinations at once.
 * The GPU kernel fills every slot, then analyzes which (if any) produced a valid match.
 */
struct Multi
{
    /** Packed input metadata: [7:4] transform, [3:0] input index */
    uint8_t inputData = 0xFF;

    /** The manufacturer key and seed used by this thread */
    Decryptor decryptor = {};

    /** Decrypted values for every learning/algorithm-type combination */
    LearningsArray decrypted = {};

    /** Index in DecryptedResults that matched, or NoMatch if none */
    KeeloqLearning::ResultIndex match = KeeloqLearning::NoMatch;

public:
    /** Which captured input (0,1,2) this result corresponds to */
    __host__ __device__ __forceinline__ uint8_t inputIndex() const { return InputInfo::Multi::getInputIndex(inputData); }

    /** Which input transform was applied */
    __host__ __device__ __forceinline__ InputsTransform inputTransform() const { return InputInfo::Multi::getTransform(inputData); }

    /** Set the captured input index */
    __host__ __device__ __forceinline__ void setInputIndex(uint8_t index) { inputData = (inputData & 0xF0) | (index & 0x0F); }

    /** Set the input transform */
    __host__ __device__ __forceinline__ void setInputTransform(InputsTransform m) { inputData = (inputData & 0x0F) | (static_cast<uint8_t>(m) << 4); }

    /** Returns an invalid/empty result sentinel */
    static Multi Invalid() { return Multi(); }

    /** True if this thread found a valid match */
    __host__ __device__ __forceinline__ bool hasMatch() const { return match != KeeloqLearning::NoMatch; }

    /** Get the matched serial number. UNSAFE! */
    __host__ __device__ __forceinline__ uint32_t matchedSerial() const
    {
        assert(hasMatch() && "Can't get matched serial from a thread result without a match");
        return decrypted.srl(match);
    }

    /** Get the matched button. UNSAFE! */
    __host__ __device__ __forceinline__ uint16_t matchedButton() const
    {
        assert(hasMatch() && "Can't get matched button from a thread result without a match");
        return decrypted.btn(match);
    }

    /** Get the matched counter. UNSAFE! */
    __host__ __device__ __forceinline__ uint16_t matchedCounter() const
    {
        assert(hasMatch() && "Can't get matched counter from a thread result without a match");
        return decrypted.cnt(match);
    }
};


/**
 * Result of a single CUDA thread in single-learning mode.
 * Stores only ONE decrypted value because the learning type is chosen at the CPU level.
 * This allows the kernel to process more decryptors per launch (smaller per-thread footprint).
 */
struct Single
{
    // Maximum number of enabled learnings that outperforms Multi mode.
    static constexpr uint8_t MaxLearningsNumInConfig = 4;

    /** Packed metadata: [7:4] transform, [3:1] input index, [0] match flag */
    uint8_t results = 0;

    /** The manufacturer key and seed used by this thread */
    Decryptor decryptor = {};

    /** Single decrypted value for the chosen learning/algorithm type */
    uint32_t decrypted = 0;

public:
    /** Extract the serial portion (10 bits). */
    __host__ __device__ __forceinline__ uint32_t srl() const { return Decode::serial(decrypted); }

    /** Extract the button portion (top 4 bits). */
    __host__ __device__ __forceinline__ uint16_t btn() const { return Decode::button(decrypted); }

    /** Extract the counter portion (lower 16 bits). */
    __host__ __device__ __forceinline__ uint16_t cnt() const { return Decode::counter(decrypted); }

    /** True if this thread found a valid match */
    __host__ __device__ __forceinline__ bool hasMatch() const { return InputInfo::Single::hasMatch(results); }

    /** Set the match flag */
    __host__ __device__ __forceinline__ void setHasMatch(uint8_t isMatch) { results = (results & 0xFE) | (isMatch & 0x01); }

    /** Set the captured input index */
    __host__ __device__ __forceinline__ void setInputIndex(uint8_t index) { results = (results & 0xF1) | ((index << 1) & 0x0E); }

    /** Which captured input (0,1,2) this result corresponds to */
    __host__ __device__ __forceinline__ uint8_t inputIndex() const { return InputInfo::Single::getInputIndex(results); }

    /** Set the input transform */
    __host__ __device__ __forceinline__ void setInputTransform(InputsTransform m) { results = (results & 0x0F) | (static_cast<uint8_t>(m) << 4); }

    /** Which input transform was applied */
    __host__ __device__ __forceinline__ InputsTransform inputsTransform() const { return InputInfo::Single::getTransform(results); }
};

} // namespace ThreadResult


/**
 * GPU-allocatable wrapper for a single enc/dec kernel result (non-bruteforce path, used for testing).
 */
struct DecryptKernelResult final : TGenericGpuObject<DecryptKernelResult>
{
    /** The single decrypted/encrypted value for the chosen learning/algorithm type */
    ThreadResult::Single result;

    /** Last CUDA error from the kernel execution, captured on read() for host-side error checking. */
    cudaError_t cudaError = cudaSuccess;

    DecryptKernelResult() : TGenericGpuObject<DecryptKernelResult>(this)
    {
    }

public:
    virtual cudaError_t read() override
    {
        cudaError = cudaGetLastError();
        CUDA_CHECK(cudaError);

        CUDA_CHECK(cudaDeviceSynchronize());

        return TGenericGpuObject<DecryptKernelResult>::read();
    }
};
