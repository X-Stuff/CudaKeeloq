#pragma once

#include "common.h"

#include <functional>
#include <cuda_runtime.h>

#include "device/kernel_input.h"
#include "device/kernel_result.h"



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


// Extern cuda kernels - Implementation are in inl.file
DECLARE_GENERATOR(GeneratorBruteforceAlphabet, KernelInput::TCudaPtr input, KernelResult::TCudaPtr resuls);
DECLARE_GENERATOR(GeneratorBruteforceFiltered, KernelInput::TCudaPtr input, KernelResult::TCudaPtr resuls);
DECLARE_GENERATOR(GeneratorBruteforceSimple, KernelInput::TCudaPtr input, KernelResult::TCudaPtr resuls);

