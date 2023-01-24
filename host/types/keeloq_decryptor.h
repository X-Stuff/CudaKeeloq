#pragma once

#include "common.h"

#include <cuda_runtime_api.h>


/**
 * Data struct which allows to decrypt encrypted data
 * In fact just (in most cases) just 64-bit integer (8 bytes array)
 */
struct Decryptor
{
	uint64_t man;

	uint32_t seed;

	Decryptor() = default;
	Decryptor(uint64_t key, uint32_t s = 0) : man(key), seed(s) {}

	__host__ __device__ inline bool operator==(const Decryptor& other) {
		return man == other.man && seed == other.seed;
	}
	__host__ __device__ inline bool operator<(const Decryptor& other) {
		return man < other.man;
	}
};