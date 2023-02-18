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
	uint64_t man;

    uint32_t seed;

	Decryptor() = default;
	Decryptor(uint64_t key, uint32_t s = 0) : man(key), seed(s), man_rev(misc::rev_bytes(man)) {}

	__host__ __device__ inline bool operator==(const Decryptor& other)
	{
		return man == other.man && seed == other.seed;
	}

	__host__ __device__ inline bool operator<(const Decryptor& other)
	{
		return man < other.man;
	}

    // Get byte-reversed manufacturer key
    __host__ __device__ inline uint64_t nam() const { return man_rev; }

    // If decryptor was initialized properly
    __host__ __device__ inline bool is_valid() const { return man != 0; }

private:

    // reversed manufacturer key
    uint64_t man_rev;
};