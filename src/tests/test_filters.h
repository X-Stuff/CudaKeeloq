#pragma once

#include "common.h"

#include "bruteforce/bruteforce_filters.h"

/**
 * Single test case for BruteforceFilters::check_filters (CPU + GPU).
 * `result` is the expected filter output; tests assert parity across host and device.
 */
struct BruteforceFiltersTestInputs
{
	uint64_t value;

	BruteforceFilters::Flags::Type flags;

	bool result;
};

namespace tests
{
	/** Validates BruteforceFilters on CPU and GPU, plus the filtered generator end-to-end. */
	bool filtersGeneration();

	/** Launches the GPU side of the filter test cases; defined in test_kernel.cu. */
	__host__ void cuda_check_bruteforce_filters(BruteforceFiltersTestInputs* tests, uint8_t num);
}
