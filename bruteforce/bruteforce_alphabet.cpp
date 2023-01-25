#include "bruteforce_alphabet.h"

#include "common.h"


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


