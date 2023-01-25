#pragma once

#include "common.h"

#include "kernels/kernel_result.h"

// It's quite heavy, forward declaration is fine
struct KeeloqKernelInput;
struct BruteforceFiltersTestInputs;

namespace Bridge
{
	extern "C" bool Kernel_CheckBridgeIsWorking();

	extern "C" bool Kernel_CheckKeeloqIsWorking();

	extern "C" void Kernel_KeeloqBruteMain(KeeloqKernelInput& inputs, KernelResult& results, uint32_t blocks, uint32_t threads);
	__forceinline KernelResult Kernel_KeeloqBruteMain(KeeloqKernelInput& inputs, uint32_t blocks, uint32_t threads)
	{
		KernelResult results;
		Kernel_KeeloqBruteMain(inputs, results, blocks, threads);
		return results;
	}
}
