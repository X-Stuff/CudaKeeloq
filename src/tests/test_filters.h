#pragma once

#include "common.h"

#include "bruteforce/bruteforce_filters.h"

struct BruteforceFiltersTestInputs
{
	uint64_t value;

	BruteforceFilters::Flags::Type flags;

	bool result;
};

namespace tests
{
	bool filters_generation();

	__host__ void cuda_check_bruteforce_filters(BruteforceFiltersTestInputs* tests, uint8_t num);
}