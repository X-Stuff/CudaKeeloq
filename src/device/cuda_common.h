#pragma once

#include <cuda_runtime_api.h>

#include "common.h"

#ifdef _MSC_VER
    #include <intrin.h>
#endif

/**
 * Bit- and byte-reversal helpers used on both host and device.
 * Device paths use hardware intrinsics; host paths fall back to compiler built-ins.
 */
namespace misc
{

__host__ __device__ __forceinline__ uint64_t rev_bytes(uint64_t x)
{
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

__host__ __device__ __forceinline__ uint64_t rev_bits(uint64_t x)
{
#if defined(__CUDA_ARCH__)
    // GPU: Hardware instruction
    return __brevll(x);
#else
    // CPU: MSVC, GCC, and Clang compatible
    x = ((x & 0x5555555555555555ULL) << 1) | ((x & 0xAAAAAAAAAAAAAAAAULL) >> 1);
    x = ((x & 0x3333333333333333ULL) << 2) | ((x & 0xCCCCCCCCCCCCCCCCULL) >> 2);
    x = ((x & 0x0F0F0F0F0F0F0F0FULL) << 4) | ((x & 0xF0F0F0F0F0F0F0F0ULL) >> 4);

    return rev_bytes(x);
#endif
}

__host__ __device__ __forceinline__ uint32_t rev_bytes32(uint32_t x)
{
#if defined(__CUDA_ARCH__)
    // GPU: Specific byte-swap intrinsic
    return __nv_bswap32(x);
#else
    // CPU: Standard GCC/Clang/MSVC intrinsic
    #ifdef _MSC_VER
        return _byteswap_ulong(x);
    #else
        return __builtin_bswap32(x);
    #endif
#endif
}

__host__ __device__ __forceinline__ uint32_t rev_bits32(uint32_t x)
{
#if defined(__CUDA_ARCH__)
    // GPU: Hardware instruction
    return __brev(x);
#else
    // Software fallback for MSVC x86/x64 (SWAR algorithm)
    x = ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1);
    x = ((x >> 2) & 0x33333333u) | ((x & 0x33333333u) << 2);
    x = ((x >> 4) & 0x0F0F0F0Fu) | ((x & 0x0F0F0F0Fu) << 4);
    x = ((x >> 8) & 0x00FF00FFu) | ((x & 0x00FF00FFu) << 8);
    return (x >> 16) | (x << 16);
#endif
}

}
