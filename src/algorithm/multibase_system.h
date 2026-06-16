#pragma once

#include <vector>

#include <cuda_runtime_api.h>

#include "common.h"

#include "algorithm/multibase_digit.h"
#include "algorithm/multibase_number.h"



/**
 * A numeric system where each digit has its own base.
 *
 * Visualise as a set of rolling cylinders installed side-by-side; each has an
 * independent alphabet. This struct is stateless, allowing it to be safely
 * shared on-device as a `const` reference for arithmetic and conversions.
 *
 * Since this is used for byte bruteforce, the maximum supported base is 255.
 *
 * e.g. NumDigits == 4
 *  So there are 4 cylinders
 *  For example 1st cylinder has base 2, second - 4, third - 6, fourth - 8
 *  So the number 1 will be equal 0001
 *  number 10 is equals 0012    (1 * 8) + 2 = | 0 | 0 | 1 | 2 | )
 *  number 100 is       0204   (12 * 8) + 4 = | 0 | 2 | 0 | 4 | )
 *  number 500 is       2224   (62 * 8) + 4 = | 2 | 2 | 2 | 4 | ) = ((10 * 6 + 2) * 8) + 4 = ((((2 * 4 + 2)) * 6 + 2) * 8) + 4
 */
template<uint8_t NumDigits = 8>
struct MultibaseSystem
{
	static_assert(NumDigits <= 8, "At the moment we only support 8 bytes numbers");

	/** Exposed count of digits in the system (useful with aliased types). */
	constexpr static uint8_t DigitsNumber = NumDigits;

	/** Per-digit alphabets (pattern usage). */
	__host__ MultibaseSystem(const std::vector<std::vector<uint8_t>>& numerals);

	/** Uniform alphabet shared across all digits (plain alphabet usage). */
	__host__ MultibaseSystem(const MultibaseDigit& digit);

	/** Default construction: every digit uses the full 0..255 byte range. */
	__host__ MultibaseSystem() : MultibaseSystem(MultibaseDigit()) { }

	// Copyable, but heavy to clone (see the constructors above) — prefer passing by reference.

public:

	/** Total count of representable numbers in this system. */
	__host__ __device__ inline size_t invariants() const;

	/** Convert a base-10 integer to the corresponding number in this system. */
	__host__ __device__ inline MultibaseNumber cast(uint64_t input) const;

	/** Add a base-10 amount to `target` (in-place) and return the reference. */
	__host__ __device__ inline MultibaseNumber& increment(MultibaseNumber& target, uint64_t amount) const;

	/** Access the configuration of a single digit by its index. */
	__host__ __device__ inline const MultibaseDigit& getConfig(uint8_t digit_index) const { assert(digit_index < NumDigits); return digits[digit_index]; }

protected:

	// the digits
	MultibaseDigit digits[NumDigits];
};

/** Convenience alias for the common 8-digit (64-bit key) case. */
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
