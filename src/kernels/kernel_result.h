#pragma once

#include "common.h"

#include "device/cuda_object.h"

/**
 *  Generic output from CUDA kernel
 */
struct KernelResult final : TGenericGpuObject<KernelResult>
{
    KernelResult() : TGenericGpuObject<KernelResult>(this)
    {
    }

    KernelResult(KernelResult&& other) noexcept : TGenericGpuObject<KernelResult>(this)
    {
        cudaError = other.cudaError;
        packed = other.packed;
    }

    KernelResult& operator=(KernelResult&& other) noexcept
    {
        cudaError = other.cudaError;
        packed = other.packed;

        SelfGpu.HOST_Ptr = this;
        return *this;
    }
public:
    static KernelResult NotStarted() { return KernelResult(); }

public:
    /**
     *  Method that kernel should call at the end of execution to update results in unified way, so we can easily read them on host side.
     */
    __host__ __device__ __forceinline__ void onKernelFinish(uint8_t numMatches)
    {
#if __CUDA_ARCH__
        // Each thread in warp writes true if it has at least 1 match
        uint32_t matchMask = __ballot_sync(0xffffffff, numMatches);

        uint32_t threadFinishedMask = __ballot_sync(0xffffffff, true);

        // Only first thread in warp updates results (AI says with 1 `if` it is faster)
        if ((threadIdx.x & (warpSize - 1)) == 0)
        {
            atomicAdd(&packed, (static_cast<uint64_t>(__popc(threadFinishedMask)) << 32) | matchMask != 0);
        }
#else
        const uint64_t addValue = (static_cast<uint64_t>(1) << 32) | static_cast<uint64_t>(numMatches != 0);
        packed += addValue;
#endif
    }

public:
    /**
     *  Flag that indicates that at least one match was found
     */
    bool hasMatch() const
    {
        return !!(packed & 0xFFFFFFFF);
    }

    /**
     *  Number of threads that successfully finished their execution, used to check if kernel executed fully or was interrupted by some error
     * You need to compare this number to number of blocks multiplied by number of threads per block,
     * to understand if kernel executed fully or was interrupted by some error (like OOM or something else)
     */
    uint32_t threadsFinished() const
    {
        return (packed >> 32);
    }

public:
    virtual void read() override
    {
        cudaError = cudaGetLastError();
        CUDA_CHECK(cudaError);

        CUDA_CHECK(cudaDeviceSynchronize());

        TGenericGpuObject<KernelResult>::read();
    }

public:
    // Additional error field that will be set on host side when read() method is called
    cudaError_t cudaError = cudaSuccess;

private:
    // Type is Clang/GNU compatibility, this is usual uint64_t
    unsigned long long int packed = 0;
};
