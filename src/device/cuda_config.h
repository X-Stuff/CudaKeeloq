#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>
#include <cmath>

#include "algorithm/keeloq/keeloq_single_result.h"
#include "algorithm/keeloq/keeloq_decryptor.h"

/**
 *  Struct holder for CUDA configuration parameters, which are used in kernel launches
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
    inline size_t total() const
    {
        return blocks * threads * substeps;
    }

public:

    /**
     *  Optimal configuration for current GPU, based on its properties and BruteforceRound's memory usage per thread.
     */
    static CudaConfig Optimal()
    {
        return { MaxCudaBlocks(), MaxCudaThreads(), 1 };
    }

    /**
     *  External API, returns GPU global memory in bytes.
     */
    static size_t MaxGlobalMemory()
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        return prop.totalGlobalMem;
    }

    static float MaxGlobalMemoryGB()
    {
        return MaxGlobalMemory() / static_cast<float>(1024 * 1024 * 1024);
    }

    /**
     *  External API, returns max number of blocks per grid for current GPU.
     */
    static uint16_t MaxCudaThreads()
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);

        return static_cast<uint16_t>(prop.maxThreadsPerBlock);
    }

    /**
     *  External API, returns calculated max number of blocks for specified memory usage per thread, and GPU properties.
     */
    static uint32_t MaxCudaBlocks()
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);

        // Usually each thread has 1 decryptor and 3 results (usually need 3 inputs)
        const auto thread_memory = prop.maxThreadsPerBlock * (sizeof(Decryptor) + sizeof(SingleResult) * 3);
        const auto max_memory = prop.totalGlobalMem;

        const auto possible_blocks = max_memory / thread_memory;
        const auto power = static_cast<int>(std::log2(possible_blocks));

        auto blocks = std::min(static_cast<uint32_t>(1 << power), static_cast<uint32_t>(prop.maxGridSize[0]));
        return blocks;
    }
};