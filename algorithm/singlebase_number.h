#pragma once

#include "common.h"

#include "algorithm/multibase_number.h"

/**
 *  Corner case when same base is used for
 * all digits. Used as alphabet
 */
struct SinglebaseNumber : public Multibase8Digits
{
	SinglebaseNumber(const std::vector<uint8_t>& numerals) : Multibase8Digits(Multibase8Digits::ByteDigit(numerals))
	{
	}

	SinglebaseNumber() : Multibase8Digits(GetDefaultByteDigit())
	{
	}

	__host__ __device__ uint8_t base_number(uint8_t index) const { return Digits[0].numeral(index); };

	__host__ __device__ uint8_t base_count() const { return Digits[0].count(); }

private:

	static Multibase8Digits::ByteDigit GetDefaultByteDigit()
	{
		static bool initialized = false;
		static std::vector<uint8_t> default_bytes(255);

		if (!initialized)
		{
			for (int i = 0; i < 255; ++i) { default_bytes[i] = i; }
			initialized = true;
		}

		static Multibase8Digits::ByteDigit default_byte_digits(default_bytes);

		return default_byte_digits;
	}
};
