#pragma once

#include "common.h"

#include "device/cuda_object.h"

/**
 * Minimal result aggregate for a kernel launch.
 *
 * Kernels call `onKernelFinish(numMatches)`; the struct atomically accumulates
 * "threads that finished" in the high 32 bits and "any match seen" in the low 32 bits,
 * so the host can read both with a single packed field.
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
    /** Zero-initialised KernelResult representing "no kernel has run yet". */
    static KernelResult NotStarted() { return KernelResult(); }

public:
    /**
     * Called by each thread at the end of the kernel; contributes its match count
     * and a "finished" signal into the packed counters.
     */
    __host__ __device__ __forceinline__ void onKernelFinish(uint32_t numMatches)
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
    /** True if at least one thread reported a match. */
    bool hasMatch() const
    {
        return !!(packed & 0xFFFFFFFF);
    }

    /**
     * Count of threads that reached `onKernelFinish`. Compare against the configured
     * (blocks x threads) to detect kernel crashes or early exits.
     */
    uint32_t threadsFinished() const
    {
        return (packed >> 32);
    }

public:
    virtual cudaError_t read() override
    {
        cudaError = cudaGetLastError();
        CUDA_CHECK(cudaError);

        CUDA_CHECK(cudaDeviceSynchronize());

        return TGenericGpuObject<KernelResult>::read();
    }

public:
    // Additional error field that will be set on host side when read() method is called
    cudaError_t cudaError = cudaSuccess;

private:
    // Type is Clang/GNU compatibility, this is usual uint64_t
    unsigned long long int packed = 0;
};
