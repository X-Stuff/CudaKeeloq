#pragma once

#include "common.h"

#include <vector>
#include <cuda_runtime_api.h>

#include "algorithm/multibase_digit.h"
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

	// For easier usage with aliased types
	constexpr static uint8_t DigitsNumber = NumDigits;

	/**
	 *  A generic case when all digits has different bases (pattern usage)
	 */
	__host__ MultibaseSystem(const std::vector<std::vector<uint8_t>>& numerals);

	/**
	 *  Use the same ByteDigit for every digit in this value (alphabet usage)
	 */
	__host__ MultibaseSystem(const MultibaseDigit& digit);

	/**
	 *  Default constructor where all Digits are Default - full range 0-255
	 */
	__host__ MultibaseSystem() : MultibaseSystem(MultibaseDigit()) { }

	// It's pretty heavy struct if you want clone it - constructor above
	// TODO: disable copy
	// MultibaseSystem(const MultibaseSystem& other) = delete;
	// MultibaseSystem& operator=(const MultibaseSystem& other) = delete;

public:

	// count of all numbers in this system
	__host__ __device__ inline size_t invariants() const;

	// cast base10 number into number of this system
	__host__ __device__ inline MultibaseNumber cast(uint64_t input) const;

	// Adds @amount in base10 to the @target argument and returns it
	__host__ __device__ inline MultibaseNumber& increment(MultibaseNumber& target, uint64_t amount) const;

	// get digit config by its index
	__host__ __device__ inline const MultibaseDigit& get_config(uint8_t digit_index) const { assert(digit_index < NumDigits); return digits[digit_index]; }

protected:

	// the digits
	MultibaseDigit digits[NumDigits];
};

//
using Multibase8DigitsSystem = MultibaseSystem<8>;


template<uint8_t NumDigits /*= 8*/>
__host__ __device__ size_t MultibaseSystem<NumDigits>::invariants() const
{
	size_t num = digits[0].size;

	UNROLL
	for (uint8_t i = 1; i < NumDigits; ++i)
	{
		num *= digits[i].size;
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
		number.indices.u8[i] = digits[i].lookup(u64Input.u8[i]);

		number.value.u8[i] = digits[i].numeral(number.indices.u8[i]);
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
		uint8_t size = digits[i].size;

		target.indices.u8[i] = static_cast<uint8_t>((amount + index) % size);
		amount = (amount + index) / size;

		target.value.u8[i] = digits[i].numeral(target.indices.u8[i]);
	}

	// here base10value will contain overflow
	// not sure what to do with it

	return target;
}


template<uint8_t NumDigits /*= 8*/>
__host__ MultibaseSystem<NumDigits>::MultibaseSystem(const std::vector<std::vector<uint8_t>>& numerals)
{
	for (uint8_t i = 0; i < NumDigits; ++i)
	{
		if (i < numerals.size())
		{
			digits[i] = MultibaseDigit(numerals[i]);
		}
		else
		{
			digits[i] = MultibaseDigit();
		}
	}
}

template<uint8_t NumDigits /*= 8*/>
__host__ MultibaseSystem<NumDigits>::MultibaseSystem(const MultibaseDigit& digit)
{
	for (uint8_t i = 0; i < NumDigits; ++i)
	{
		digits[i] = digit;
	}
}
