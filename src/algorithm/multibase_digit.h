#pragma once

#include "common.h"

#include <vector>
#include <string>


/**
 *  This struct represents single digit in multi-base system (each digit has own base)
 * Imagine this like cylinder with N elements on it
 *
 * This struct represents all possible variant for a number (byte_ in multi-base system (attack pattern) setup
 */
struct MultibaseDigit
{
	template<uint8_t TNum> friend struct MultibaseSystem;

	// Creates digit config
	__host__ inline MultibaseDigit(const std::vector<uint8_t>& numerals);

public:
	// Return numeral by index
	__host__ __device__ inline uint8_t numeral(uint8_t in_index) const;

	// Cast from byte-255 value to Digit's config
	__host__ __device__ inline uint8_t cast(uint8_t value) const { return numeral(lookup(value)); }

	// return numeral index of the value
	// e.g.
	//  num = { 0x04, 0xAB, 0xd7, 0x56 }
	//  lookup(0xAB) returns 1
	//  lookup(0x56) returns 3
	//  lookup(0xFF) returns 0 (not found returns default)
	__host__ __device__ inline uint8_t lookup(uint8_t value) const { return lut[value]; }

	// return count of possible numerals for that digit
	__host__ __device__ inline uint8_t count() const { return size; }

	__host__ inline std::vector<uint8_t> as_vector() const { return std::vector<uint8_t>(&num[0], &num[0] + size); }

	__host__ inline std::string to_string() const;

private:
	MultibaseDigit() : MultibaseDigit(DefaultByteArray<>::as_vector<std::vector<uint8_t>>())
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

__host__ inline std::string MultibaseDigit::to_string() const
{
    std::string hex;

	for (int i = 0; i < count(); ++i)
	{
        std::string fmt(i == 0 ? "%X" : ":%X");
        hex += str::format<std::string>(fmt, numeral(i));
	}

	return hex;
}
