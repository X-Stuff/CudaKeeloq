#pragma once

#include "common.h"

#include <vector>
#include <string>

#include <cuda_runtime_api.h>


NS_LOCATION_BEGIN

/**
 *
 */
struct BruteforceAlphabet
{
	BruteforceAlphabet() = default;

	// Duplicates will be removed
	BruteforceAlphabet(const std::vector<uint8_t>& alphabet);

	// size of the alphabet
	__host__ __device__ inline size_t size() const { return num; }

	// Number of all possible combinations for this alphabet
	__host__ __device__ inline size_t invariants() const { return (size_t)pow(num, sizeof(uint64_t)); }

	// return index of value (cannot fail - if value not in a LUT - always return 0 index)
	__host__ __device__ inline uint8_t lookup(uint8_t value) const { return lut[value]; }

	// return value by index
	__host__ __device__ inline uint8_t operator[](uint8_t index) { return alp[index]; }

	// Since alphabet represents its own number system - you may wasy to add numbers
	__host__ __device__ uint64_t add(uint64_t number, uint64_t value) const;

	// Since alphabet represents its own number system - you may wasy to add numbers
	__host__ __device__ void add(uint8_t number[sizeof(uint64_t)], uint64_t value) const;

	// lookup for each byte
	__host__ __device__ uint64_t lookup(uint64_t value) const;

	__host__ __device__ uint64_t value(uint64_t index) const;

	__host__ std::string toString() const;

private:

	static const uint16_t Size = 0xFF + 1; // 256

	// The alphabet itself (256 bytes max)
	uint8_t alp[Size] = { 0 };

	// The alphabet lookup table
	uint8_t lut[Size] = { 0 };

	// Actual size of alphabet
	uint8_t num = 0;
};


NS_LOCATION_END