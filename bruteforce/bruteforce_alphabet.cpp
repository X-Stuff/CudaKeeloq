#include "bruteforce_alphabet.h"

__host__ std::string BruteforceAlphabet::toString() const
{
	char tmp[255 * 3] = { 0 }; // one byte is 'XX:' last is XX\0
	int write_index = 0;
	for (int i = 0; i < system.num_digits(); ++i)
	{
		write_index += sprintf_s(&tmp[write_index], sizeof(tmp) - write_index, i == 0 ? "%X" : ":%X", system.get_digit(i));
	}
	return std::string(tmp);
}


