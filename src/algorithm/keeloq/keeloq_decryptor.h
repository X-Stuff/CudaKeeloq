#pragma once

#include "common.h"

#include <cuda_runtime_api.h>

#include "device/cuda_common.h"


/**
 * Data struct which allows to decrypt encrypted data
 * In fact just (in most cases) just 64-bit integer (8 bytes array)
 */
struct Decryptor
{
    Decryptor() = default;

    __host__ __device__ Decryptor(uint64_t k, uint32_t s) : key(k), key_seed(s), key_rev(misc::rev_bytes(key)) {}

	__host__ __device__ inline bool operator==(const Decryptor& other)
	{
		return key == other.key && key_seed == other.key_seed;
	}

	__host__ __device__ inline bool operator<(const Decryptor& other)
	{
		return key < other.key;
	}

    // Get manufacturer key
    __host__ __device__ inline uint64_t man() const { return key; }

    // Get seed
    __host__ __device__ inline uint32_t seed() const { return key_seed; }

    // Get byte-reversed manufacturer key
    __host__ __device__ inline uint64_t nam() const { return key_rev; }

    // If decryptor was initialized properly
    __host__ __device__ inline bool is_valid() const { return key != 0; }

protected:

    // manufacturer key
    uint64_t key;

    // seed (for special learning types only)
    uint32_t key_seed;

    // reversed manufacturer key
    uint64_t key_rev;

};