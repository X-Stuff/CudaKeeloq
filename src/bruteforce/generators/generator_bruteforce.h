#pragma once

#include <cuda_runtime.h>

#include "common.h"

#include "device/cuda_config.h"
#include "kernels/kernel_result.h"

#include "algorithm/keeloq/keeloq_kernel_input.h"


/**
 * Declares a host-side wrapper around a cu-file generator kernel.
 * Pairs each generator struct with its __global__ kernel symbol and an extern "C" getter.
 */
#define DECLARE_GENERATOR(name, ...) \
    extern "C" void* GENERATOR_KERNEL_GETTER_NAME(name)(); \
    __global__ void GENERATOR_KERNEL_NAME(name)(__VA_ARGS__); \
    struct name : public IGenerator<name> \
    {\
        typedef void(*func)(__VA_ARGS__); \
        inline static func GetKernelFunctionPtr() { return (func)GENERATOR_KERNEL_GETTER_NAME(name)(); } \
    };


/**
 * CRTP helper that launches a generator kernel using its derived class's function pointer.
 */
template<typename TSelf>
struct IGenerator
{
    typedef void(*KernelFunc)(KeeloqKernelInput::TCudaPtr input, KernelResult::TCudaPtr results);

    static inline void LaunchKernel(const CudaConfig& cuda, KeeloqKernelInput::TCudaPtr input, KernelResult::TCudaPtr results)
    {
        void* args[] = { &input, &results };

        auto* func = TSelf::GetKernelFunctionPtr();

        auto error = cudaLaunchKernel((const void*)func, dim3(cuda.blocks), dim3(cuda.threads), args, 0, nullptr);
        CUDA_CHECK(error);
    }
};

/**
 * Dispatcher that picks the correct generator kernel for a given BruteforceConfig and
 * writes the next batch of decryptors to GPU memory.
 */
struct GeneratorBruteforce
{
    /**
     * Generate the next batch of decryptors on the GPU based on the active bruteforce type.
     *
     * Note: `inputs.decryptors.size()` must be `N * blocks * threads` with N > 0.
     */
    static cudaError_t PrepareDecryptors(KeeloqKernelInput& inputs, const CudaConfig& cuda);
};


// Extern cuda kernels - Implementation are in inl.file
DECLARE_GENERATOR(GeneratorBruteforcePattern, KeeloqKernelInput::TCudaPtr input, KernelResult::TCudaPtr resuls);
DECLARE_GENERATOR(GeneratorBruteforceFiltered, KeeloqKernelInput::TCudaPtr input, KernelResult::TCudaPtr resuls);
DECLARE_GENERATOR(GeneratorBruteforceSimple, KeeloqKernelInput::TCudaPtr input, KernelResult::TCudaPtr resuls);
DECLARE_GENERATOR(GeneratorBruteforceSeed, KeeloqKernelInput::TCudaPtr input, KernelResult::TCudaPtr resuls);
