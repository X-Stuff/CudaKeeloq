#pragma once

#include <cmath>
#include <cstdint>

#include <cuda_runtime_api.h>

#include "algorithm/keeloq/keeloq_decryptor.h"
#include "algorithm/keeloq/keeloq_single_result.h"

/**
 * Launch configuration for the bruteforce kernels (grid/block dimensions and per-thread work).
 * Also exposes static helpers that query device capabilities and pick sensible defaults.
 */
struct CudaConfig
{
    // Number of blocks to use in CUDA kernel launches
    uint32_t blocks = 0;

    // Number of threads per block to use in CUDA kernel launches
    uint16_t threads = 0;

    // Number of substeps per thread in CUDA kernel launches (not really used, always 1)
    uint16_t substeps = 1;

public:
    /** Total per-launch work units (blocks × threads × substeps). */
    inline uint32_t total() const
    {
        return blocks * threads * substeps;
    }

    /** Total thread count across the grid (blocks × threads). */
    inline uint32_t threadsCount() const
    {
        return blocks * threads;
    }

public:

    /** Best-effort default configuration based on the device's memory and grid limits. */
    static CudaConfig Optimal()
    {
        static CudaConfig optimal = { MaxCudaBlocks(), MaxCudaThreads(), 1 };
        return optimal;
    }

    /** Minimal configuration used by the unit tests (single block, max threads). */
    static CudaConfig Tests()
    {
        static CudaConfig tests = { 1, MaxCudaThreads(), 1 };
        return tests;
    }

    /** GPU global memory, in bytes. */
    static size_t MaxGlobalMemory()
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        return prop.totalGlobalMem;
    }

    /** GPU global memory, in GB. */
    static float MaxGlobalMemoryGB()
    {
        return MaxGlobalMemory() / static_cast<float>(1024 * 1024 * 1024);
    }

    /** Maximum threads per block supported by the current device. */
    static uint16_t MaxCudaThreads()
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);

        return static_cast<uint16_t>(prop.maxThreadsPerBlock);
    }

    /** Maximum block count that fits in the device's memory and grid limits for this workload. */
    static uint32_t MaxCudaBlocks(int threadsPerBlock = 0, size_t sizeofResult = 0)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);

        threadsPerBlock = threadsPerBlock > 0 ? threadsPerBlock : prop.maxThreadsPerBlock;
        sizeofResult = sizeofResult > 0 ? sizeofResult : sizeof(SingleResult);

        // Usually each thread has 1 decryptor and 3 results (usually need 3 inputs)
        const auto thread_memory = threadsPerBlock * (sizeof(Decryptor) + sizeofResult * 3);
    #if _DEBUG
        // Limit to 5GB in debug
        static constexpr size_t MaxMeoryInDebug = (1024 * 1024 * 1024) * static_cast<size_t>(5);
        const auto max_memory = std::min(prop.totalGlobalMem, MaxMeoryInDebug);
    #else
        const auto max_memory = prop.totalGlobalMem;
    #endif

        const auto possible_blocks = max_memory / thread_memory;
        const auto power = static_cast<int>(std::log2(possible_blocks));

        auto blocks = std::min(static_cast<uint32_t>(1 << power), static_cast<uint32_t>(prop.maxGridSize[0]));
        return blocks;
    }
};
