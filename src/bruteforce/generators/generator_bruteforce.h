#pragma once

#include "common.h"

#include <cuda_runtime.h>

#include "algorithm/keeloq/keeloq_kernel_input.h"
#include "kernels/kernel_result.h"


/**
 * Declare new struct which represents a wrapper around cu implementation
 */
#define DECLARE_GENERATOR(name, ...) \
	extern "C" void* GENERATOR_KERNEL_GETTER_NAME(name)(); \
	__global__ void GENERATOR_KERNEL_NAME(name)(__VA_ARGS__); \
	struct name : public IGenerator<name> \
	{\
		typedef void(*func)(__VA_ARGS__); \
		inline static func GetKernelFunctionPtr() { return (func)GENERATOR_KERNEL_GETTER_NAME(name)(); } \
	};


template<typename TSelf>
struct IGenerator
{
	typedef void(*KernelFunc)(KeeloqKernelInput::TCudaPtr input, KernelResult::TCudaPtr results);

	static inline void LaunchKernel(uint16_t blocks, uint16_t threads, KeeloqKernelInput::TCudaPtr input, KernelResult::TCudaPtr results)
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
	static int PrepareDecryptors(KeeloqKernelInput& inputs, uint16_t blocks, uint16_t threads);
};


// Extern cuda kernels - Implementation are in inl.file
DECLARE_GENERATOR(GeneratorBruteforcePattern, KeeloqKernelInput::TCudaPtr input, KernelResult::TCudaPtr resuls);
DECLARE_GENERATOR(GeneratorBruteforceFiltered, KeeloqKernelInput::TCudaPtr input, KernelResult::TCudaPtr resuls);
DECLARE_GENERATOR(GeneratorBruteforceSimple, KeeloqKernelInput::TCudaPtr input, KernelResult::TCudaPtr resuls);

