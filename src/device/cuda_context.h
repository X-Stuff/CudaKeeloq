#pragma once

#include "common.h"

#include <device_launch_parameters.h>


// For loop macro
// indexer @i Will be incremented by amount of all threads in the context
#define CUDA_FOR_THREAD_ID(ctx, i, count) for(uint32_t i = ctx.thread_id; i < count; i += ctx.thread_max)

struct CudaContext
{
    uint32_t thread_max;

    uint32_t thread_id;

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