#pragma once

#include <string>
#include <vector>

#include "common.h"


/**
 * Represents a single digit in a multi-base number where each digit has its own base.
 *
 * Conceptually a "rolling cylinder" whose face shows one numeral from a fixed set.
 * Used to describe per-byte search alphabets for the bruteforce attack patterns.
 */
struct MultibaseDigit
{
	template<uint8_t TNum> friend struct MultibaseSystem;

	/** Build a digit from an explicit set of numerals (max 256). */
	__host__ inline MultibaseDigit(const std::vector<uint8_t>& numerals);

public:
	/** Return the numeral stored at the given position in this digit's alphabet. */
	__host__ __device__ inline uint8_t numeral(uint8_t in_index) const;

	/** Map a raw byte value to the configured numeral for it. */
	__host__ __device__ inline uint8_t cast(uint8_t value) const { return numeral(lookup(value)); }

	/**
	 * Return the numeral index for a given byte value.
	 * e.g. num = { 0x04, 0xAB, 0xd7, 0x56 }: lookup(0xAB) == 1, lookup(0x56) == 3,
	 * lookup(0xFF) == 0 (missing value falls back to default 0).
	 */
	__host__ __device__ inline uint8_t lookup(uint8_t value) const { return lut[value]; }

	/** Number of distinct numerals configured for this digit (its base). */
	__host__ __device__ inline uint8_t count() const { return size; }

	/** Copy of the digit's numerals as a host vector. */
	__host__ inline std::vector<uint8_t> asVector() const { return std::vector<uint8_t>(&num[0], &num[0] + size); }

	/** Colon-separated hex string of the digit's numerals (debug/printing). */
	__host__ inline std::string toString() const;

private:
	MultibaseDigit() : MultibaseDigit(DefaultByteArray<>::asVector<std::vector<uint8_t>>())
	{
	}

	// numeral values. it may be not just 0,1,2,3,4...
	// but for base 4 it may be: 0xA3, 0xCC, 0x01, 0x22
	uint8_t num[256] = { 0 };

	// lookup table:
	//  at index that equals numeral value there is a value which represents index in numerals
	//  e.g.                              https://asciiflow.com/
	//                                  ┌───────────────────────┐
	//                                  ▲                       │
	// numerals = [ 0x03, 0x02, 0x01, 0x00, ... garbage. ]      │
	//                                  ▲                       │
	//               ┌──────────────────┘                       │
	//               ▲                                          │
	//	    lut = [ 0x03, 0x02, 0x01, 0x00, 0x00 ... 0x00]      │
	//               ▲                                          │
	//               └──────────────────────────────────────────┘
	//
	uint8_t lut[256] = { 0 };

	// Actual size of numerals (the base if number representing by this digit)
	uint8_t size = 0;
};

__host__ __device__ uint8_t MultibaseDigit::numeral(uint8_t in_index) const
{
	assert(in_index < size);

	// Important for optimization purposes
	// WE ARE NOT USING (value % size)
	return num[in_index];
}

__host__ inline MultibaseDigit::MultibaseDigit(const std::vector<uint8_t>& numerals)
{
	// incrementing in the loop, for the duplicate numerals cases
	size = 0;

	for (uint8_t i = 0; i < numerals.size(); ++i)
	{
		// Getting next numeral candidate
		uint8_t numeral_value = numerals[i];

		// if there is 0 value (index) in the lookup table
		// that means `numeral_value` wasn't added to available values yet
		if (!lut[numeral_value])
		{
			// putting index of `numeric_value` to the lut
			lut[numeral_value] = size;

			// setting the size-th numeral
			num[size] = numeral_value;

			// increasing the size
			++size;
		}
	}

	assert(size > 0 && "Digit base should be at least 0");
}

__host__ inline std::string MultibaseDigit::toString() const
{
    std::string hex;

	for (int i = 0; i < count(); ++i)
	{
        std::string fmt(i == 0 ? "%X" : ":%X");
        hex += str::format<std::string>(fmt, numeral(i));
	}

	return hex;
}
