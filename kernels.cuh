#pragma once

#include "common.h"

#include "keeloq_types.cuh"

namespace Tests
{
	void Kernel_RunBruteforceFiltersTests(BruteforceFilters::Test::Inputs* tests, uint8_t num, uint32_t blocks = 1, uint32_t threads = 1);
}

namespace Generators
{
	void Kernel_GenerateDecryptorsFiltered(uint32_t blocks, uint32_t threads, KernelInput::TCudaPtr input, KernelResult::TCudaPtr results);

	void Kernel_GenerateDecryptorsAlphabet(uint32_t blocks, uint32_t threads, KernelInput::TCudaPtr input, KernelResult::TCudaPtr results);
}
