#pragma once

#include "common.h"

#include <stdint.h>

#include "device/cuda_object.h"
#include "algorithm/keeloq/keeloq_learning_types.h"
#include "algorithm/keeloq/keeloq_encrypted.h"
#include "algorithm/keeloq/keeloq_decryptor.h"


/**
 *  Result of single keeloq run (encryption or decryption).
 *
 * EncParcel - is input for decryption as OTA, for encryption is should be created in specific way
 * Decryptor - is used decryptor for this result, contains manufacturer key and seed (if needed)
 * LearningsArray - is array of decrypted or encrypted values for each learning type (now summary 18 possible combinations)
 */
struct SingleResult
{
    /**
     *  Per-Learning with modifiers type array of decrypted or encrypted values.
     *
     * e.g.:
     *  at index 0 - decrypted result for Simple learning type with Normal modifier
     *  at index 1 - decrypted result for Simple learning type with Inverted modifier
     *  etc.
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
        __host__ __device__ __forceinline__ uint32_t srl(uint8_t index) const
        {
            return ((*this)[index] >> 16) & 0x3ff;
        }

        __host__ __device__ __forceinline__ uint32_t btn(uint8_t index) const
        {
            return ((*this)[index] >> 28);
        }

        __host__ __device__ __forceinline__ uint32_t cnt(uint8_t index) const
        {
            return ((*this)[index] & 0x0000FFFF);
        }

        void print(uint8_t element, uint64_t ota, bool ismatch) const;

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
 *  Specific result for single decryption kernel.
 */
struct DecryptKernelResult final : TGenericGpuObject<DecryptKernelResult>
{
    SingleResult result;

    DecryptKernelResult() : TGenericGpuObject<DecryptKernelResult>(this)
    {
    }
};