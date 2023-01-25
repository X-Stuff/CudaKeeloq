#pragma once

#include "common.h"

#include "device/kernel_result.h"

// It's quite heavy, forward declaration is fine
struct KernelInput;
struct BruteforceFiltersTestInputs;

namespace Bridge
{
	extern "C" bool Kernel_CheckBridgeIsWorking();

	extern "C" bool Kernel_CheckKeeloqIsWorking();

	extern "C" void Kernel_KeeloqBruteMain(KernelInput& inputs, KernelResult& results, uint32_t blocks, uint32_t threads);
	__forceinline KernelResult Kernel_KeeloqBruteMain(KernelInput& inputs, uint32_t blocks, uint32_t threads)
	{
		KernelResult results;
		Kernel_KeeloqBruteMain(inputs, results, blocks, threads);
		return results;
	}

	extern "C" void Kernel_LaunchFiltersTests(BruteforceFiltersTestInputs* tests, uint8_t num);
}
