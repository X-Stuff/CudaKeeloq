#pragma once

#include "common.h"

#include "bruteforce/bruteforce_filters.h"

namespace Tests
{
	bool FiltersGeneration();
}

struct BruteforceFiltersTestInputs
{
	uint64_t value;

	BruteforceFilters::Flags::Type flags;

	bool result;
};