#include "bruteforce_alphabet.h"

#include "unroll.h"

#include <assert.h>

NS_LOCATION_BEGIN

BruteforceAlphabet::BruteforceAlphabet(const std::vector<uint8_t>& alphabet)
{
	num = 0;

	for (int i = 0; i < alphabet.size(); ++i)
	{
		uint8_t byte = alphabet[i];
		if (!lut[byte])
		{
			lut[byte] = num;
			alp[num] = byte;
			++num;
		}
	}

	assert(num < BruteforceAlphabet::Size && "Using all bytes values as alphabet is not efficient");
}

__host__ __device__ uint64_t BruteforceAlphabet::add(uint64_t number, uint64_t value) const
{
	uint8_t result[8] = {};
	*(uint64_t*)result = number;

	add(result, value);

	return *(uint64_t*)result;
}

__host__ __device__ void BruteforceAlphabet::add(uint8_t number[sizeof(uint64_t)], uint64_t value) const
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

__host__ __device__ uint64_t BruteforceAlphabet::lookup(uint64_t value) const
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

__host__ __device__ uint64_t BruteforceAlphabet::value(uint64_t index) const
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

__host__ std::string BruteforceAlphabet::toString() const
{
	char tmp[255 * 3] = { 0 }; // one byte is 'XX:' last is XX\0
	int write_index = 0;
	for (int i = 0; i < num; ++i)
	{
		write_index += sprintf_s(&tmp[write_index], sizeof(tmp) - write_index, i == 0 ? "%X" : ":%X", alp[i]);
	}
	return std::string(tmp);
}

NS_LOCATION_END