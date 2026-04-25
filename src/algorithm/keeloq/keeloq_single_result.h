#pragma once

#include "common.h"

#include <stdint.h>

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

        __host__ __device__ inline uint32_t operator[](uint32_t index) const
        {
            assert(index < KeeloqLearning::DecryptedResults::Size && "Invalid index of decrypted data. Bigger than last element");
#if __CUDA_ARCH__
            return __ldca(&data[index]);
#else
            return data[index];
#endif
        }
        __host__ __device__ inline uint32_t at(KeeloqLearning::Type learning, KeeloqLearning::Mod mod) const
        {
            const auto index = KeeloqLearning::Matrix::getIndex(learning, mod);
            return (*this)[index];
        }

        __host__ __device__ inline uint32_t srl(KeeloqLearning::Type learning, KeeloqLearning::Mod mod) const
        {
            return (at(learning, mod) >> 16) & 0x3ff;
        }

        __host__ __device__ inline uint32_t btn(KeeloqLearning::Type learning, KeeloqLearning::Mod mod) const
        {
            return (at(learning, mod) >> 28);
        }

        __host__ __device__ inline uint32_t cnt(KeeloqLearning::Type learning, KeeloqLearning::Mod mod) const
        {
            return (at(learning, mod) & 0x0000FFFF);
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
    KeeloqLearning::MatchIndex match = KeeloqLearning::NoMatch;

	void print(bool onlymatch = true) const;
};