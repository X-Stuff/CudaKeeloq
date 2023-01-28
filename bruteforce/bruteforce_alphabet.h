#pragma once

#include "common.h"

#include <vector>
#include <string>

#include <cuda_runtime_api.h>

#include "algorithm/singlebase_number.h"


/**
 *
 */
struct BruteforceAlphabet_old
{
	BruteforceAlphabet_old() = default;

	// Duplicates will be removed
	BruteforceAlphabet_old(const std::vector<uint8_t>& alphabet);

	// size of the alphabet
	__host__ __device__ inline size_t size() const { return num; }

	// Number of all possible combinations for this alphabet
	__host__ __device__ inline size_t invariants() const { return (size_t)pow(num, sizeof(uint64_t)); }

	// return index of value (cannot fail - if value not in a LUT - always return 0 index)
	__host__ __device__ inline uint8_t lookup(uint8_t value) const { return lut[value]; }

	// return value by index
	__host__ __device__ inline uint8_t operator[](uint8_t index) { return alp[index]; }

	// Since alphabet represents its own number system - you may want to add numbers
	__host__ __device__ inline uint64_t add(uint64_t number, uint64_t value) const;

	// Since alphabet represents its own number system - you may want to add numbers
	__host__ __device__ inline void add(uint8_t number[sizeof(uint64_t)], uint64_t value) const;

	// lookup for each byte
	__host__ __device__ inline uint64_t lookup(uint64_t value) const;

	__host__ __device__ inline uint64_t value(uint64_t index) const;

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

__host__ __device__ inline uint64_t BruteforceAlphabet_old::add(uint64_t number, uint64_t value) const
{
	uint8_t result[8] = {};
	*(uint64_t*)result = number;

	add(result, value);

	return *(uint64_t*)result;
}

__host__ __device__ inline void BruteforceAlphabet_old::add(uint8_t number[sizeof(uint64_t)], uint64_t value) const
{
	UNROLL
	for (int i = 0; i < sizeof(uint64_t); ++i)
	{
		uint8_t digit = value % num;                // 17 % 6 = 5
		uint16_t addition = number[i] + digit;      // 5 + 5 = 10 (6 + 4)
		number[i] = addition % num;                 // n[i] = 4

		value /= num;                               // 17 / 6 = 2

		uint8_t carry = addition >= num;            // 10 > 6
		value += carry;                             // 2 + 1 = 3
	}
}

__host__ __device__ inline uint64_t BruteforceAlphabet_old::value(uint64_t index) const
{
	uint8_t* pIndex = (uint8_t*)&index;
	uint8_t result[8] = {
		alp[pIndex[0]],
		alp[pIndex[1]],
		alp[pIndex[2]],
		alp[pIndex[3]],
		alp[pIndex[4]],
		alp[pIndex[5]],
		alp[pIndex[6]],
		alp[pIndex[7]],
	};
	return *(uint64_t*)result;
}

__host__ __device__ uint64_t BruteforceAlphabet_old::lookup(uint64_t value) const
{
	uint64_t result = 0;
	uint8_t* pResult = (uint8_t*)&result;
	uint8_t* pValue = (uint8_t*)&value;

	UNROLL
	for (uint8_t i = 0; i < sizeof(uint64_t); ++i)
	{
		// Valid or 0 (first letter in alphabet)
		pResult[i] = lookup(pValue[i]);
	}
	return result;
}



struct BruteforceAlphabet_new
{
	BruteforceAlphabet_new(const std::vector<uint8_t>& alphabet)
		: number_repesentation(alphabet)
	{
	}

	BruteforceAlphabet_new()
	{
	}

	// Set start value for generation round
	__host__ void set_start(uint64_t value) { number_repesentation.set(value); }

	// Return underlying singlebase number
	__host__ __device__ const SinglebaseNumber& number() const { return number_repesentation; }

	// will find first valid SingleBase number for input base10 number
	// e.g.
	//  number's base is: { 0x11, 0x22, 0x33 }
	//  number's value is not important
	//  base10 number is: 0x11 22 11 BB 10 23 33 33
	//  the result will : 0x11 22 11 11 11 11 33 33
	//		0xbb 0x10 0x23 were not found - thus replaced with first numeral
	__host__ __device__ uint64_t cast(uint64_t base10number) const { return number_repesentation.convert(base10number); }

	__host__ __device__ size_t size() const { return number_repesentation.base_count(); }

	__host__ std::string toString() const;

private:

	SinglebaseNumber number_repesentation;
};


using BruteforceAlphabet = BruteforceAlphabet_new;