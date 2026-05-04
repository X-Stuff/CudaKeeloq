#pragma once

#include <vector>

#include <cuda_runtime_api.h>

#include "common.h"


/**
 * Packed 64-bit value accessible either as a single uint64 or as an 8-byte array.
 * Used by multi-base arithmetic to manipulate per-byte digits without shifts.
 */
union U64Number
{
	uint64_t u64;

	uint8_t u8[8];
};


/**
 * A number expressed in a multi-base number system.
 * Stores both the per-digit numeral indices and the resolved byte value.
 * Constructed exclusively through MultibaseSystem (the "alphabet").
 */
struct MultibaseNumber
{
	template<uint8_t TNum> friend struct MultibaseSystem;

	/** Resolved 64-bit byte value of this number. */
	__host__ __device__ uint64_t number() const { return value.u64; }

private:

	U64Number value   = {0};

	U64Number indices = {0};
};
