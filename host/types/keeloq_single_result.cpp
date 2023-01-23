#include "keeloq_single_result.h"

#include <stdio.h>


void SingleResult::DecryptedArray::print(uint8_t element, bool ismatch) const
{
	printf("[%-40s] Btn:0x%X\tSerial:0x%X\tCounter:0x%X\t%s\n", KeeloqLearningType::Name(element),
		(data[element] >> 28),              // Button
		(data[element] >> 16) & 0x3ff,      // Serial
		data[element] & 0xFFFF,             // Counter
		(ismatch ? "(MATCH)" : ""));
}

void SingleResult::DecryptedArray::print() const
{
	for (uint8_t i = 0; i < ResultsCount; ++i)
	{
		print(i, false);
	}
}

void SingleResult::print(bool onlymatch /* = true */) const
{
	printf("Results (Input: 0x%llX - Man key: 0x%llX)\n\n", ota, man);

	for (uint8_t i = 0; i < ResultsCount; ++i)
	{
		bool isMatch = match == i;
		if (!onlymatch)
		{
			results.print(i, isMatch);
		}
		else if (isMatch)
		{
			results.print(i, isMatch);
		}
	}
	printf("\n");
}