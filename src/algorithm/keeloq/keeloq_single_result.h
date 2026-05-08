#pragma once

#include <cstdint>

#include "common.h"

#include "device/cuda_object.h"

#include "algorithm/keeloq/keeloq_decryptor.h"
#include "algorithm/keeloq/keeloq_encrypted.h"
#include "algorithm/keeloq/keeloq_learning_types.h"


/**
 * Outcome of a single keeloq encryption or decryption run.
 * Holds the source decryptor, the per-learning decrypted values, and the match index (if any).
 */
struct SingleResult
{
    /**
     * Per-learning result storage, indexed by `DecryptedResults::getIndex(learning, mods)`.
     * Provides CUDA-tuned accessors (srl/btn/cnt) to unpack the decrypted 32-bit value.
     */
	struct LearningsArray
	{
		// fixed side array for every learning type
        // If is in global memory (common case) - use operator[] - though cache
        // If is thread local - use direct access
        KeeloqLearning::DecryptedResults data;

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
            return ((*this)[index] >> 16) & 0x3ff;
        }

        /** Extract the button portion of a decrypted entry (top 4 bits). */
        __host__ __device__ __forceinline__ uint16_t btn(uint8_t index) const
        {
            return ((*this)[index] >> 28);
        }

        /** Extract the counter portion of a decrypted entry (lower 16 bits). */
        __host__ __device__ __forceinline__ uint16_t cnt(uint8_t index) const
        {
            return ((*this)[index] & 0x0000FFFF);
        }

        /** Extract the serial number from value (lower 16 bits). */
        __host__ __device__ __forceinline__ uint32_t static serial(uint32_t value)
        {
            return (value >> 16) & 0x3ff;
        }

        /** Extract the serial number from value (lower 16 bits). */
        __host__ __device__ __forceinline__ uint8_t static button(uint32_t value)
        {
            return (value >> 28);
        }

        /** Extract the counter portion from value (lower 16 bits). */
        __host__ __device__ __forceinline__ uint16_t static counter(uint32_t value)
        {
            return (value & 0x0000FFFF);
        }

        /** Pretty-print one decoded entry, highlighting the matched slot. */
        void print(const KeeloqLearning::LearningItem& item, uint32_t srl, KeeloqLearning::ResultIndex match) const;

        /** Pretty-print every valid decoded entry. */
        void print() const;
    };

    // Input encrypted data
    uint8_t inputIndex = 0xFF;

    // used manufacturer key and seed for this result
    Decryptor decryptor = {};

    // Processed values for each known learning type (decrypted or encrypted depending on the call), indexed by learning type and modifier
	LearningsArray decrypted = {};

	// Index in array that represents pairs of learning types and modes.
    // Set by GPU after analysis if there was a match
    KeeloqLearning::ResultIndex match = KeeloqLearning::NoMatch;

public:
    static SingleResult Invalid() { return SingleResult(); }

    /** True if this result has match to inputs with the internal decryptor */
    bool hasMatch() const { return match != KeeloqLearning::NoMatch; }

    /** Prints decrypted results, all learnings (may be multiple matches) */
	void print(const std::vector<EncParcel>& inputs) const;
};


/**
 * Result wrapper for the single-run enc/dec kernel (non-bruteforce path).
 */
struct DecryptKernelResult final : TGenericGpuObject<DecryptKernelResult>
{
    SingleResult result;

    DecryptKernelResult() : TGenericGpuObject<DecryptKernelResult>(this)
    {
    }
};
