#pragma once

#include "common.h"

#include <stdint.h>

#include "device/cuda_object.h"
#include "algorithm/keeloq/keeloq_learning_types.h"
#include "algorithm/keeloq/keeloq_encrypted.h"
#include "algorithm/keeloq/keeloq_decryptor.h"


/**
 *  For each testing manufacturer key we retrieve this results
 * Depending on selected learning type you may have from 1 to 16 (now it the last)
 * decrypted results for further analysis
 */
struct SingleResult
{
	struct DecryptedArray
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
    EncParcel encrypted = {};

    // used manufacturer key and seed for this result
    Decryptor decryptor = {};

	// Decrypted values for each known learning type
	DecryptedArray decrypted = {};

	// Index in array that represents pairs of learning types and modes.
    // Set by GPU after analysis if there was a match
    KeeloqLearning::ResultIndex match = KeeloqLearning::NoMatch;

	void print(bool onlymatch = true) const;
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