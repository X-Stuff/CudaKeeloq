#pragma once

#include <device_launch_parameters.h>

#include "common.h"


// For loop macro
// indexer @i Will be incremented by amount of all threads in the context
#define CUDA_FOR_THREAD_ID(ctx, i, count) for(uint32_t i = ctx.thread_id; i < count; i += ctx.thread_max)

/**
 * Lightweight view of the current CUDA execution context — the global thread id
 * and the total thread count. Built from CUDA builtins on device, inert on host.
 */
struct CudaContext
{
    // Maximum overall threads
    uint32_t thread_max;

    // Global thread id (within the entire grid, blockIdx.x * blockDim.x + threadIdx.x)
    uint32_t thread_id;

    /** Populates a CudaContext from the current kernel's thread/block indices. */
    __host__ __device__ static inline CudaContext Get()
    {
    #if __CUDA_ARCH__
        return CudaContext
        {
            gridDim.x * blockDim.x,
            blockIdx.x * blockDim.x + threadIdx.x
        };
    #else
        assert(false && "CUDA context is not available on host");
        return {};
    #endif
    }
};
