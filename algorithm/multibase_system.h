#pragma once

#include "common.h"

#include <vector>
#include <cuda_runtime_api.h>

#include "algorithm/multibase_number.h"


/**
*  This represents a number system where each digit has it's one base
* Imagine this as set of rolling cylinders, set side-by-side.
* This is stateless structure. It allows to do arithmetical calculations
* and conversions.
*
* Since application of this struct is only byte bruteforce - the maximum supported base is 255
*
* e.g. NumDigits == 4
*  So there are 4 cylinders
*  For example 1st cylinder has base 2, second - 4, third - 6, fourth - 8
*  So the number 1 will be equal 0001
*  number 10 is equals 0012    (1 * 8) + 2 = | 0 | 0 | 1 | 2 | )
*  number 100 is       0204   (12 * 8) + 4 = | 0 | 2 | 0 | 4 | )
*  number 500 is       2224   (62 * 8) + 4 = | 2 | 2 | 2 | 4 | ) = ((10 * 6 + 2) * 8) + 4 = ((((2 * 4 + 2)) * 6 + 2) * 8) + 4
*
*  Since this structure is pretty heavy
* The idea is NOT allow non-const methods on device
* You should have a single const reference on device.
*/
template<uint8_t NumDigits = 8>
struct MultibaseSystem
{
	static_assert(NumDigits <= 8, "At the moment we only support 8 bytes numbers");

	/**
	 *  A generic case when all digits has different bases
	 */
	__host__ MultibaseSystem(const std::vector<uint8_t> numerals[NumDigits]);

	// It's pretty heavy struct if you want clone it - constructor above
	// TODO: disable copy
	// MultibaseNumber(const MultibaseNumber& other) = delete;
	// MultibaseNumber& operator=(const MultibaseNumber& other) = delete;

protected:

	struct DigitConfig;

	/**
	 *  Use the same ByteDigit for every digit in this value
	 */
	__host__ MultibaseSystem(const DigitConfig& digit);

public:

	// count of all numbers in this system
	__host__ __device__ inline size_t invariants() const;

	// cast base10 number into number of this system
	__host__ __device__ inline MultibaseNumber cast(uint64_t input) const;

	// Adds @amount in base10 to the @target argument and returns it
	__host__ __device__ inline MultibaseNumber& increment(MultibaseNumber& target, uint64_t amount) const;

protected:

	// Imagine this as one cylinder
	struct DigitConfig
	{
		friend struct MultibaseSystem<NumDigits>;

		// Creates digit
		__host__ DigitConfig(const std::vector<uint8_t>& numerals);

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

	private:
		DigitConfig() = default;

		// numeral values. it may be not just 0,1,2,3,4...
		// but for base 4 it may be: 0xA3, 0xCC, 0x01, 0x22
		uint8_t num[0xFF] = {0};

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
		uint8_t lut[0xFF] = {0};

		// Actual size of numerals (the base if number representing by this digit)
		uint8_t size = 0;
	};

protected:

	// the digits
	DigitConfig Digits[NumDigits];
};

template<uint8_t NumDigits /*= 8*/>
__host__ __device__ size_t MultibaseSystem<NumDigits>::invariants() const
{
	size_t num = Digits[0].size;

	UNROLL
	for (uint8_t i = 1; i < NumDigits; ++i)
	{
		num *= Digits[i].size;
	}

	return num;
}


template<uint8_t NumDigits /*= 8*/>
__host__ __device__ MultibaseNumber MultibaseSystem<NumDigits>::cast(uint64_t input) const
{
	MultibaseNumber number;
	U64Number u64Input = { input };

	UNROLL
	for (uint8_t i = 0; i < NumDigits; ++i)
	{
		number.indices.u8[i] = Digits[i].lookup(u64Input.u8[i]);

		number.value.u8[i] = Digits[i].numeral(number.indices.u8[i]);
	}

	return number;
}

template<uint8_t NumDigits /*= 8*/>
__host__ __device__ inline MultibaseNumber& MultibaseSystem<NumDigits>::increment(MultibaseNumber& target, uint64_t amount) const
{
	UNROLL
	for (uint8_t i = 0; i < NumDigits; ++i)
	{
		uint8_t index = target.indices.u8[i];
		uint8_t size = Digits[i].size;

		target.indices.u8[i] = static_cast<uint8_t>((amount + index) % size);
		amount = (amount + index) / size;

		target.value.u8[i] = Digits[i].numeral(target.indices.u8[i]);
	}

	// here base10value will contain overflow
	// not sure what to do with it

	return target;
}

template<uint8_t NumDigits /*= 8*/>
__host__ __device__ uint8_t MultibaseSystem<NumDigits>::DigitConfig::numeral(uint8_t in_index) const
{
	assert(in_index < size);

	// Important for optimization purposes
	// WE ARE NOT USING (value % size)
	return num[in_index];
}

template<uint8_t NumDigits /*= 8*/>
__host__ MultibaseSystem<NumDigits>::MultibaseSystem(const std::vector<uint8_t> numerals[NumDigits])
{
	for (uint8_t i = 0; i < NumDigits; ++i)
	{
		Digits[i] = ByteDigit(numerals[i]);
	}
}

template<uint8_t NumDigits /*= 8*/>
__host__ MultibaseSystem<NumDigits>::MultibaseSystem(const DigitConfig& digit)
{
	for (uint8_t i = 0; i < NumDigits; ++i)
	{
		Digits[i] = digit;
	}
}


template<uint8_t NumDigits /*= 8*/>
__host__ MultibaseSystem<NumDigits>::DigitConfig::DigitConfig(const std::vector<uint8_t>& numerals)
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
