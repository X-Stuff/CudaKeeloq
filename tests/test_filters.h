#pragma once

#include "common.h"

#include "host/types/bruteforce_filters.h"

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