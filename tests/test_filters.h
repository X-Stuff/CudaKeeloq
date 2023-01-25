#pragma once

#include "common.h"

#include "bruteforce/bruteforce_filters.h"

struct BruteforceFiltersTestInputs
{
	uint64_t value;

	BruteforceFilters::Flags::Type flags;

	bool result;
};

namespace Tests
{
	bool FiltersGeneration();
}

extern "C" void Kernel_LaunchFiltersTests(BruteforceFiltersTestInputs * tests, uint8_t num);