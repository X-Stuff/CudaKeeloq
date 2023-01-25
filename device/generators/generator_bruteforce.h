#pragma once

#include "common.h"

#include <functional>
#include <cuda_runtime.h>

#include "device/kernel_input.h"
#include "device/kernel_result.h"

USE_NS_LOCATION


template<typename TSelf>
struct IGenerator
{
	typedef void(*KernelFunc)(KernelInput::TCudaPtr input, KernelResult::TCudaPtr results);

	static inline void LaunchKernel(uint16_t blocks, uint16_t threads, KernelInput::TCudaPtr input, KernelResult::TCudaPtr results)
	{
		void* args[] = { &input, &results };

		auto* func = TSelf::GetKernelFunctionPtr();

		auto error = cudaLaunchKernel((const void*)func, dim3(blocks), dim3(threads), args, 0, nullptr);
		CUDA_CHECK(error);
	}
};

struct GeneratorBruteforce
{
	// Checks type of used generator in inputs and launches kernel to generate next batch of decryptors
	// Decryptors are generated on GPU and stored in GPU memory
	static int PrepareDecryptors(KernelInput& inputs, uint16_t blocks, uint16_t threads);
};