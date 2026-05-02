#pragma once

#include "common.h"

#include <cuda_runtime_api.h>

#ifdef _MSC_VER
    #include <intrin.h>
#endif

namespace misc
{
__host__ __device__ __forceinline__ uint64_t rev_bits(uint64_t x) {
#if defined(__CUDA_ARCH__)
    // GPU: Hardware instruction
    return __brevll(x);
#else
    // CPU: MSVC, GCC, and Clang compatible
    x = ((x & 0x5555555555555555ULL) << 1) | ((x & 0xAAAAAAAAAAAAAAAAULL) >> 1);
    x = ((x & 0x3333333333333333ULL) << 2) | ((x & 0xCCCCCCCCCCCCCCCCULL) >> 2);
    x = ((x & 0x0F0F0F0F0F0F0F0FULL) << 4) | ((x & 0xF0F0F0F0F0F0F0F0ULL) >> 4);

    #ifdef _MSC_VER
        return _byteswap_uint64(x);
    #else
        return __builtin_bswap64(x);
    #endif
#endif
}

__host__ __device__ __forceinline__ uint64_t rev_bytes(uint64_t x) {
#if defined(__CUDA_ARCH__)
    // GPU: Specific byte-swap intrinsic
    return __nv_bswap64(x);
#else
    // CPU: Standard GCC/Clang/MSVC intrinsic
    #ifdef _MSC_VER
        return _byteswap_uint64(x);
    #else
        return __builtin_bswap64(x);
    #endif
#endif
}

}